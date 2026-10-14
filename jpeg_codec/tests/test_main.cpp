// Self-contained unit tests (no external framework). Run with:  make test
//
// What is checked, bottom-up:
//   DCT        fast separable version == textbook double sum; inverse; Parseval; DC of flat block
//   zig-zag    is a permutation, starts the way JPEG's does
//   quant      table scaling rules and bounds
//   rle        category / amplitude round trip, ZRL / EOB behaviour
//   huffman    Kraft equality, prefix-freeness, textbook example, length limiting, table I/O
//   bitio      random round trip
//   codec      lossless entropy stage, odd sizes, quality behaviour, determinism, corrupt input
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "jlc/bitio.hpp"
#include "jlc/codec.hpp"
#include "jlc/dct.hpp"
#include "jlc/huffman.hpp"
#include "jlc/image.hpp"
#include "jlc/metrics.hpp"
#include "jlc/quant.hpp"
#include "jlc/rle.hpp"
#include "jlc/zigzag.hpp"

using namespace jlc;

// ---------------------------------------------------------------------------
// Mini test framework
// ---------------------------------------------------------------------------
static int g_failures = 0;
static int g_checks = 0;
static std::vector<std::pair<std::string, std::function<void()>>>& registry() {
    static std::vector<std::pair<std::string, std::function<void()>>> r;
    return r;
}
struct Registrar {
    Registrar(const char* name, std::function<void()> f) { registry().push_back({name, std::move(f)}); }
};
#define TEST(name)                                  \
    static void name();                             \
    static Registrar reg_##name(#name, name);       \
    static void name()
#define CHECK(cond)                                                                   \
    do {                                                                              \
        ++g_checks;                                                                   \
        if (!(cond)) {                                                                \
            ++g_failures;                                                             \
            std::printf("    FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
        }                                                                             \
    } while (0)

// Deterministic pseudo random numbers.
struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed) {}
    uint32_t next() {
        s = s * 1664525u + 1013904223u;
        return s >> 8;
    }
    int range(int lo, int hi) { return lo + static_cast<int>(next() % static_cast<uint32_t>(hi - lo + 1)); }
    double uniform() { return next() / 16777216.0; }
};

static Image randomImage(int w, int h, uint32_t seed) {
    Rng r(seed);
    Image img(w, h);
    for (auto& p : img.pixels) p = static_cast<uint8_t>(r.range(0, 255));
    return img;
}

// ---------------------------------------------------------------------------
// DCT
// ---------------------------------------------------------------------------
TEST(dct_matches_textbook_definition) {
    Rng r(1);
    for (int trial = 0; trial < 50; ++trial) {
        double x[64], fast[64], slow[64];
        for (double& v : x) v = r.uniform() * 255.0 - 128.0;
        forwardDCT8x8(x, fast);
        forwardDCT8x8Naive(x, slow);
        double maxDiff = 0;
        for (int i = 0; i < 64; ++i) maxDiff = std::max(maxDiff, std::fabs(fast[i] - slow[i]));
        CHECK(maxDiff < 1e-10);
    }
}

TEST(dct_inverse_recovers_input) {
    Rng r(2);
    for (int trial = 0; trial < 50; ++trial) {
        double x[64], y[64], back[64];
        for (double& v : x) v = r.uniform() * 255.0 - 128.0;
        forwardDCT8x8(x, y);
        inverseDCT8x8(y, back);
        double maxDiff = 0;
        for (int i = 0; i < 64; ++i) maxDiff = std::max(maxDiff, std::fabs(x[i] - back[i]));
        CHECK(maxDiff < 1e-10);
    }
}

TEST(dct_matrix_is_orthonormal) {
    const double* C = dctMatrix();
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            double dot = 0;
            for (int k = 0; k < 8; ++k) dot += C[i * 8 + k] * C[j * 8 + k];
            CHECK(std::fabs(dot - (i == j ? 1.0 : 0.0)) < 1e-12);
        }
    }
}

TEST(dct_preserves_energy_parseval) {
    Rng r(3);
    double x[64], y[64];
    for (double& v : x) v = r.uniform() * 255.0 - 128.0;
    forwardDCT8x8(x, y);
    double ex = 0, ey = 0;
    for (int i = 0; i < 64; ++i) {
        ex += x[i] * x[i];
        ey += y[i] * y[i];
    }
    CHECK(std::fabs(ex - ey) < 1e-8 * ex);
}

TEST(dct_flat_block_has_only_dc_equal_to_8_times_value) {
    double x[64], y[64];
    for (double& v : x) v = 37.0;
    forwardDCT8x8(x, y);
    CHECK(std::fabs(y[0] - 8.0 * 37.0) < 1e-10);
    for (int i = 1; i < 64; ++i) CHECK(std::fabs(y[i]) < 1e-10);
}

// ---------------------------------------------------------------------------
// Zig-zag and quantization tables
// ---------------------------------------------------------------------------
TEST(zigzag_is_permutation_and_has_jpeg_prefix) {
    const auto& z = zigzagOrder();
    std::vector<int> seen(64, 0);
    for (int v : z) {
        CHECK(v >= 0 && v < 64);
        if (v >= 0 && v < 64) ++seen[v];
    }
    for (int c : seen) CHECK(c == 1);
    const int expected[] = {0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5};
    for (int i = 0; i < 16; ++i) CHECK(z[i] == expected[i]);
    CHECK(z[63] == 63);
}

TEST(quant_table_scaling_rules) {
    const QTable q50 = makeQuantTable(50);
    CHECK(q50 == baseLuminanceTable());
    const QTable q100 = makeQuantTable(100);
    for (int v : q100) CHECK(v == 1);
    const QTable q1 = makeQuantTable(1);
    for (int v : q1) CHECK(v == 255);
    // Lower quality never gives a smaller step anywhere.
    for (int q = 1; q < 100; ++q) {
        const QTable a = makeQuantTable(q), b = makeQuantTable(q + 1);
        for (int i = 0; i < 64; ++i) {
            CHECK(a[i] >= b[i]);
            CHECK(a[i] >= 1 && a[i] <= 255);
        }
    }
    CHECK(makeQuantTable(-5) == makeQuantTable(1));
    CHECK(makeQuantTable(500) == makeQuantTable(100));
}

TEST(quantize_rounds_half_up) {
    CHECK(quantizeValue(7.9, 16) == 0);
    CHECK(quantizeValue(8.0, 16) == 1);   // exactly 0.5 rounds up
    CHECK(quantizeValue(-8.0, 16) == 0);  // -0.5 rounds up to 0
    CHECK(quantizeValue(-8.1, 16) == -1);
    CHECK(quantizeValue(100.0, 10) == 10);
    // Deterministic tie-break: "almost exactly" .5 from either side rounds the same way.
    CHECK(roundHalfUp(0.5 - 1e-13) == 1.0);
    CHECK(roundHalfUp(0.5 + 1e-13) == 1.0);
    CHECK(roundHalfUp(-0.5 - 1e-13) == 0.0);
    CHECK(roundHalfUp(0.5 - 1e-6) == 0.0);  // a real difference is not swallowed
}

// ---------------------------------------------------------------------------
// Run-length tokens
// ---------------------------------------------------------------------------
TEST(category_and_amplitude_round_trip) {
    CHECK(category(0) == 0);
    CHECK(category(1) == 1);
    CHECK(category(-1) == 1);
    CHECK(category(2) == 2);
    CHECK(category(3) == 2);
    CHECK(category(4) == 3);
    CHECK(category(-1023) == 10);
    CHECK(category(1024) == 11);
    for (int v = -2047; v <= 2047; ++v) {
        const int s = category(v);
        const uint16_t bits = amplitudeBits(v, s);
        CHECK(bits < (1u << std::max(s, 1)));
        CHECK(extendAmplitude(bits, s) == v);
    }
    // spot check against the JPEG convention
    CHECK(amplitudeBits(-1, 1) == 0);
    CHECK(amplitudeBits(1, 1) == 1);
    CHECK(amplitudeBits(-3, 2) == 0);
    CHECK(amplitudeBits(-2, 2) == 1);
    CHECK(amplitudeBits(2, 2) == 2);
    CHECK(amplitudeBits(3, 2) == 3);
}

TEST(tokenizer_emits_zrl_and_eob_correctly) {
    int16_t zz[64] = {};
    zz[0] = 5;
    zz[1] = 3;
    zz[20] = -2;  // 18 zeros between index 1 and 20 -> ZRL + run 2
    std::vector<Token> t;
    int prev = 0;
    tokenizeBlock(zz, prev, t);
    CHECK(prev == 5);
    CHECK(t.size() == 5);
    if (t.size() == 5) {
        CHECK(t[0].isAC == 0 && t[0].symbol == 3);       // DC diff 5 -> category 3
        CHECK(t[1].isAC == 1 && t[1].symbol == 0x02);    // run 0, size 2 (value 3)
        CHECK(t[2].isAC == 1 && t[2].symbol == kZRL);    // 16 zeros
        CHECK(t[3].isAC == 1 && t[3].symbol == 0x22);    // run 2, size 2 (value -2)
        CHECK(t[4].isAC == 1 && t[4].symbol == kEOB);    // everything after is zero
    }
}

TEST(tokenizer_omits_eob_when_last_coefficient_nonzero) {
    int16_t zz[64] = {};
    zz[63] = 1;
    std::vector<Token> t;
    int prev = 0;
    tokenizeBlock(zz, prev, t);
    // DC (diff 0), then ZRL x3 (48 zeros), then run 14 / size 1 -> no EOB
    CHECK(t.size() == 1 + 3 + 1);
    CHECK(t.back().symbol == ((14 << 4) | 1));
}

TEST(tokenizer_dc_is_differential) {
    int16_t a[64] = {}, b[64] = {};
    a[0] = 10;
    b[0] = 7;
    std::vector<Token> t;
    int prev = 0;
    tokenizeBlock(a, prev, t);
    tokenizeBlock(b, prev, t);
    // block a: DC diff +10 (category 4); block b: diff -3 (category 2)
    CHECK(t[0].symbol == 4);
    CHECK(t[2].symbol == 2);
    CHECK(extendAmplitude(t[2].amp, 2) == -3);
}

// ---------------------------------------------------------------------------
// Huffman
// ---------------------------------------------------------------------------
static bool isPrefixFree(const HuffTable& t) {
    const int n = t.alphabetSize();
    for (int a = 0; a < n; ++a) {
        if (!t.length(a)) continue;
        for (int b = 0; b < n; ++b) {
            if (a == b || !t.length(b) || t.length(b) < t.length(a)) continue;
            if ((t.code(b) >> (t.length(b) - t.length(a))) == t.code(a)) return false;
        }
    }
    return true;
}

TEST(huffman_textbook_example) {
    // CLRS: a:45 b:13 c:12 d:16 e:9 f:5  ->  lengths 1,3,3,3,4,4
    const auto len = huffmanCodeLengths({45, 13, 12, 16, 9, 5});
    CHECK(len[0] == 1);
    CHECK(len[1] == 3);
    CHECK(len[2] == 3);
    CHECK(len[3] == 3);
    CHECK(len[4] == 4);
    CHECK(len[5] == 4);
    HuffTable t(len);
    CHECK(isPrefixFree(t));
}

TEST(huffman_kraft_equality_and_optimal_cost) {
    Rng r(5);
    for (int trial = 0; trial < 20; ++trial) {
        std::vector<uint64_t> f(256, 0);
        int used = 0;
        for (auto& v : f) {
            if (r.range(0, 3) == 0) continue;
            v = static_cast<uint64_t>(r.range(1, 100000));
            ++used;
        }
        const auto len = huffmanCodeLengths(f);
        long double kraft = 0;
        double cost = 0, total = 0;
        for (size_t s = 0; s < f.size(); ++s) {
            if (f[s] == 0) { CHECK(len[s] == 0); continue; }
            CHECK(len[s] >= 1);
            kraft += std::ldexp(1.0L, -len[s]);
            cost += static_cast<double>(f[s]) * len[s];
            total += static_cast<double>(f[s]);
        }
        if (used >= 2) CHECK(std::fabs(static_cast<double>(kraft) - 1.0) < 1e-12);  // full binary tree
        // Entropy bound: average length < H + 1
        double H = 0;
        for (uint64_t v : f) if (v) H -= (v / total) * std::log2(v / total);
        CHECK(cost / total >= H - 1e-9);
        CHECK(cost / total < H + 1.0);
        CHECK(isPrefixFree(HuffTable(len)));
    }
}

TEST(huffman_edge_cases) {
    CHECK(huffmanCodeLengths({0, 0, 0}) == std::vector<uint8_t>({0, 0, 0}));
    CHECK(huffmanCodeLengths({0, 7, 0}) == std::vector<uint8_t>({0, 1, 0}));
    CHECK(huffmanCodeLengths({3, 3}) == std::vector<uint8_t>({1, 1}));
}

TEST(huffman_length_limit_is_enforced) {
    // Fibonacci frequencies force a maximally skewed tree (depth ~ number of symbols).
    std::vector<uint64_t> f(45);
    f[0] = 1;
    f[1] = 1;
    for (size_t i = 2; i < f.size(); ++i) f[i] = f[i - 1] + f[i - 2];
    const auto len = huffmanCodeLengths(f, 32);
    CHECK(*std::max_element(len.begin(), len.end()) <= 32);
    long double kraft = 0;
    for (uint8_t l : len) kraft += std::ldexp(1.0L, -l);
    CHECK(kraft <= 1.0L + 1e-12L);
    HuffTable t(len);
    CHECK(isPrefixFree(t));
    // And the limited code must still round-trip every symbol.
    std::vector<uint8_t> bytes;
    BitWriter bw(bytes);
    for (int s = 0; s < 45; ++s) bw.put(t.code(s), t.length(s));
    bw.flush();
    BitReader br(bytes.data(), bytes.size());
    for (int s = 0; s < 45; ++s) CHECK(t.decodeSymbol(br) == s);
}

TEST(huffman_table_serialization_round_trip) {
    Rng r(8);
    std::vector<uint64_t> f(256, 0);
    for (int i = 0; i < 256; ++i) if (i % 7 != 3) f[i] = static_cast<uint64_t>(r.range(1, 5000));
    const HuffTable a(huffmanCodeLengths(f));
    std::vector<uint8_t> bytes;
    a.serialize(bytes);
    size_t pos = 0;
    const HuffTable b = HuffTable::deserialize(bytes.data(), bytes.size(), pos, 256);
    CHECK(pos == bytes.size());
    for (int s = 0; s < 256; ++s) {
        CHECK(a.length(s) == b.length(s));
        CHECK(a.code(s) == b.code(s));
    }
}

// ---------------------------------------------------------------------------
// Bit I/O
// ---------------------------------------------------------------------------
TEST(bitio_random_round_trip) {
    Rng r(9);
    std::vector<std::pair<uint32_t, int>> items;
    std::vector<uint8_t> bytes;
    BitWriter bw(bytes);
    for (int i = 0; i < 5000; ++i) {
        const int n = r.range(0, 32);
        const uint32_t v = n == 0 ? 0u : (r.next() * 2654435761u) & (n == 32 ? 0xFFFFFFFFu : ((1u << n) - 1u));
        items.push_back({v, n});
        bw.put(v, n);
    }
    bw.flush();
    BitReader br(bytes.data(), bytes.size());
    for (auto& [v, n] : items) CHECK(br.bits(n) == v);
}

// ---------------------------------------------------------------------------
// Whole codec
// ---------------------------------------------------------------------------
TEST(entropy_stage_is_exactly_lossless) {
    // For many images and qualities: decoding the bitstream gives back precisely the
    // quantized coefficients that the encoder produced.
    const int sizes[][2] = {{8, 8}, {13, 9}, {64, 64}, {101, 77}};
    uint32_t seed = 1;
    for (auto& sz : sizes) {
        for (int q : {1, 10, 50, 90, 100}) {
            const Image img = (seed % 2) ? randomImage(sz[0], sz[1], seed) : makeSyntheticImage(sz[0], sz[1], seed);
            ++seed;
            const QuantizedImage a = analyze(img, q);
            const auto bytes = encode(img, q);
            const QuantizedImage b = decodeCoefficients(bytes.data(), bytes.size());
            CHECK(a.zigzag == b.zigzag);
            CHECK(b.width == img.width && b.height == img.height && b.quality == q);
        }
    }
}

TEST(round_trip_keeps_dimensions_for_odd_sizes) {
    const int sizes[][2] = {{1, 1}, {1, 20}, {20, 1}, {7, 9}, {8, 8}, {17, 13}, {64, 64}, {100, 75}, {257, 129}};
    for (auto& sz : sizes) {
        const Image img = makeSyntheticImage(sz[0], sz[1], 3);
        const Image back = decode(encode(img, 75));
        CHECK(back.width == sz[0] && back.height == sz[1]);
        CHECK(back.pixels.size() == img.pixels.size());
    }
}

TEST(flat_images_survive_exactly_at_high_quality) {
    for (int v : {0, 1, 77, 128, 200, 255}) {
        Image img(40, 24);
        std::fill(img.pixels.begin(), img.pixels.end(), static_cast<uint8_t>(v));
        const Image back = decode(encode(img, 100));
        CHECK(back.pixels == img.pixels);
    }
}

TEST(quality_100_error_is_tiny_and_bounded) {
    // With every quantizer step equal to 1, the only error is rounding of coefficients
    // (each off by at most 0.5) plus the final rounding to integer pixels.
    for (uint32_t seed : {1u, 2u, 3u}) {
        const Image img = randomImage(96, 96, seed);
        const Image back = decode(encode(img, 100));
        CHECK(maxAbsError(img, back) <= 3);
        CHECK(mse(img, back) < 0.5);
        const Image smooth = makeSyntheticImage(96, 96, seed);
        const Image back2 = decode(encode(smooth, 100));
        CHECK(maxAbsError(smooth, back2) <= 3);
        CHECK(mse(smooth, back2) < 0.5);
    }
}

TEST(quality_trades_size_for_fidelity_monotonically) {
    const Image img = makeSyntheticImage(160, 160, 4);
    size_t prevBytes = 0;
    double prevPsnr = 0;
    for (int q : {5, 15, 30, 50, 70, 85, 95}) {
        const auto bytes = encode(img, q);
        const double p = psnr(img, decode(bytes));
        CHECK(bytes.size() >= prevBytes);
        CHECK(p >= prevPsnr - 1e-9);
        prevBytes = bytes.size();
        prevPsnr = p;
    }
}

TEST(extreme_dc_jumps_round_trip) {
    // Alternating black / white 8x8 blocks give the largest possible DC differences.
    Image img(64, 64);
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x) img.at(x, y) = (((x / 8) + (y / 8)) % 2) ? 255 : 0;
    const auto bytes = encode(img, 100);
    const QuantizedImage a = analyze(img, 100);
    const QuantizedImage b = decodeCoefficients(bytes.data(), bytes.size());
    CHECK(a.zigzag == b.zigzag);
    CHECK(maxAbsError(img, decode(bytes)) <= 3);
}

TEST(fused_and_staged_paths_are_bit_identical) {
    // The fast "fused" execution strategy must change speed only, never the result.
    const int sizes[][2] = {{1, 1}, {9, 7}, {64, 64}, {131, 77}, {200, 200}};
    uint32_t seed = 20;
    for (auto& sz : sizes) {
        for (int q : {1, 25, 50, 90, 100}) {
            const Image img = (seed % 2) ? randomImage(sz[0], sz[1], seed) : makeSyntheticImage(sz[0], sz[1], seed);
            ++seed;
            const auto staged = encode(img, q, nullptr, false);
            const auto fused = encode(img, q, nullptr, true);
            CHECK(staged == fused);
            const Image a = decode(staged, nullptr, false);
            const Image b = decode(staged, nullptr, true);
            CHECK(a.pixels == b.pixels);
        }
    }
}

TEST(encoding_is_deterministic) {
    const Image img = makeSyntheticImage(120, 90, 6);
    CHECK(encode(img, 60) == encode(img, 60));
}

TEST(coefficient_range_is_within_theory) {
    // Level-shifted pixels are in [-128, 127]; the orthonormal 8x8 DCT gives
    // |DC| <= 8*128 = 1024, and every AC coefficient is bounded by ~ 1024 as well.
    const Image img = randomImage(128, 128, 11);
    const QuantizedImage qi = analyze(img, 100);  // Q = 1 everywhere, so values are raw coefficients
    int maxAbs = 0;
    for (int16_t v : qi.zigzag) maxAbs = std::max(maxAbs, std::abs(static_cast<int>(v)));
    CHECK(maxAbs <= 1024);
    for (size_t b = 0; b < qi.zigzag.size() / 64; ++b) CHECK(std::abs(qi.zigzag[b * 64]) <= 1024);
}

TEST(malformed_streams_are_rejected_not_crashed) {
    const Image img = makeSyntheticImage(48, 48, 7);
    const auto good = encode(img, 50);

    std::vector<uint8_t> bad = good;
    bad[0] = 'X';
    bool threw = false;
    try { decode(bad); } catch (const std::exception&) { threw = true; }
    CHECK(threw);

    for (size_t cut : {size_t(0), size_t(5), size_t(12), size_t(40), good.size() / 2, good.size() - 1}) {
        std::vector<uint8_t> trunc(good.begin(), good.begin() + static_cast<long>(cut));
        threw = false;
        try { decode(trunc); } catch (const std::exception&) { threw = true; }
        CHECK(threw);
    }

    // Random corruption of the payload must never crash (it may or may not throw).
    Rng r(10);
    for (int i = 0; i < 200; ++i) {
        std::vector<uint8_t> corrupt = good;
        for (int k = 0; k < 4; ++k) corrupt[static_cast<size_t>(r.range(13, static_cast<int>(good.size()) - 1))] ^= static_cast<uint8_t>(1u << r.range(0, 7));
        try { decode(corrupt); } catch (const std::exception&) {}
    }
    CHECK(true);
}

// ---------------------------------------------------------------------------
int main() {
    for (auto& [name, fn] : registry()) {
        const int before = g_failures;
        fn();
        std::printf("[%s] %s\n", g_failures == before ? " ok " : "FAIL", name.c_str());
    }
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
