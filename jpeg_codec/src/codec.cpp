#include "jlc/codec.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

#include "jlc/bitio.hpp"
#include "jlc/dct.hpp"
#include "jlc/huffman.hpp"
#include "jlc/quant.hpp"
#include "jlc/rle.hpp"
#include "jlc/zigzag.hpp"

namespace jlc {

namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

constexpr char kMagic[4] = {'J', 'L', 'C', '1'};
constexpr size_t kHeaderSize = 13;
constexpr uint64_t kMaxPixels = 1ull << 30;  // sanity limit for corrupt headers

void putU32(std::vector<uint8_t>& out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

uint32_t getU32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

int clampQuality(int q) { return q < 1 ? 1 : (q > 100 ? 100 : q); }

}  // namespace

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

QuantizedImage analyze(const Image& img, int quality, EncodeTimings* timings, bool fused) {
    if (img.width <= 0 || img.height <= 0 || img.pixels.size() != static_cast<size_t>(img.width) * img.height) {
        throw std::invalid_argument("analyze: invalid image");
    }
    quality = clampQuality(quality);

    QuantizedImage qi;
    qi.width = img.width;
    qi.height = img.height;
    qi.quality = quality;
    qi.blocksX = (img.width + 7) / 8;
    qi.blocksY = (img.height + 7) / 8;
    const size_t numBlocks = static_cast<size_t>(qi.blocksX) * qi.blocksY;

    const QTable q = makeQuantTable(quality);
    const auto& zz = zigzagOrder();
    qi.zigzag.resize(numBlocks * 64);

    // Copy one 8x8 block out of the image, replicating edge pixels beyond the border
    // (padding) and subtracting 128 (level shift) so the values are centred around 0.
    auto gatherBlock = [&](int bx, int by, double* block) {
        for (int r = 0; r < 8; ++r) {
            const int y = std::min(by * 8 + r, img.height - 1);  // replicate last row
            for (int c = 0; c < 8; ++c) {
                const int x = std::min(bx * 8 + c, img.width - 1);  // replicate last column
                block[r * 8 + c] = static_cast<double>(img.at(x, y)) - 128.0;
            }
        }
    };
    // Quantize one block of DCT coefficients and store it in zig-zag order.
    auto quantizeBlock = [&](const double* in, int16_t* out) {
        for (int i = 0; i < 64; ++i) {
            const int natural = zz[i];
            out[i] = static_cast<int16_t>(quantizeValue(in[natural], q[natural]));
        }
    };

    if (fused) {
        // One pass, everything per block in a 64-double local buffer.
        auto t0 = Clock::now();
        for (int by = 0; by < qi.blocksY; ++by) {
            for (int bx = 0; bx < qi.blocksX; ++bx) {
                double block[64], coef[64];
                gatherBlock(bx, by, block);
                forwardDCT8x8(block, coef);
                quantizeBlock(coef, &qi.zigzag[(static_cast<size_t>(by) * qi.blocksX + bx) * 64]);
            }
        }
        if (timings) timings->dct_ms = msSince(t0);  // DCT + quantization together
        return qi;
    }

    // ---- Stage 1: pad (edge replication), level shift (-128), forward DCT of every block.
    auto t0 = Clock::now();
    std::vector<double> coeffs(numBlocks * 64);
    for (int by = 0; by < qi.blocksY; ++by) {
        for (int bx = 0; bx < qi.blocksX; ++bx) {
            double block[64];
            gatherBlock(bx, by, block);
            forwardDCT8x8(block, &coeffs[(static_cast<size_t>(by) * qi.blocksX + bx) * 64]);
        }
    }
    if (timings) timings->dct_ms = msSince(t0);

    // ---- Stage 2: quantize and reorder into zig-zag order.
    t0 = Clock::now();
    for (size_t b = 0; b < numBlocks; ++b) quantizeBlock(&coeffs[b * 64], &qi.zigzag[b * 64]);
    if (timings) timings->quant_zigzag_ms = msSince(t0);
    return qi;
}

std::vector<uint8_t> encode(const Image& img, int quality, EncodeTimings* timings, bool fused) {
    const auto tStart = Clock::now();
    EncodeTimings local;
    EncodeTimings* tm = timings ? timings : &local;
    *tm = EncodeTimings{};

    QuantizedImage qi = analyze(img, quality, tm, fused);
    const size_t numBlocks = static_cast<size_t>(qi.blocksX) * qi.blocksY;

    // ---- Stage 3: DC prediction + run-length tokenization, count symbol frequencies.
    auto t0 = Clock::now();
    std::vector<Token> tokens;
    tokens.reserve(numBlocks * 8);
    int prevDC = 0;
    for (size_t b = 0; b < numBlocks; ++b) tokenizeBlock(&qi.zigzag[b * 64], prevDC, tokens);
    std::vector<uint64_t> dcFreq(kDcAlphabet, 0), acFreq(kAcAlphabet, 0);
    for (const Token& t : tokens) (t.isAC ? acFreq : dcFreq)[t.symbol]++;
    tm->scan_ms = msSince(t0);

    // ---- Stage 4: build the two Huffman tables from the symbol statistics.
    t0 = Clock::now();
    const HuffTable dcTable(huffmanCodeLengths(dcFreq));
    const HuffTable acTable(huffmanCodeLengths(acFreq));
    tm->huff_build_ms = msSince(t0);

    // ---- Stage 5: serialize header, tables and the entropy-coded payload.
    t0 = Clock::now();
    std::vector<uint8_t> out;
    out.reserve(kHeaderSize + 128 + tokens.size());
    out.insert(out.end(), kMagic, kMagic + 4);
    putU32(out, static_cast<uint32_t>(img.width));
    putU32(out, static_cast<uint32_t>(img.height));
    out.push_back(static_cast<uint8_t>(qi.quality));
    dcTable.serialize(out);
    acTable.serialize(out);
    {
        BitWriter bw(out);
        for (const Token& t : tokens) {
            const HuffTable& table = t.isAC ? acTable : dcTable;
            bw.put(table.code(t.symbol), table.length(t.symbol));
            bw.put(t.amp, t.nbits);
        }
        bw.flush();
    }
    tm->bitpack_ms = msSince(t0);
    tm->total_ms = msSince(tStart);
    return out;
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

QuantizedImage decodeCoefficients(const uint8_t* data, size_t size) {
    if (size < kHeaderSize || std::memcmp(data, kMagic, 4) != 0) {
        throw std::runtime_error("decode: not a JLC1 stream");
    }
    const uint32_t width = getU32(data + 4);
    const uint32_t height = getU32(data + 8);
    const int quality = data[12];
    if (width == 0 || height == 0 || static_cast<uint64_t>(width) * height > kMaxPixels) {
        throw std::runtime_error("decode: invalid image dimensions");
    }
    if (quality < 1 || quality > 100) throw std::runtime_error("decode: invalid quality");

    size_t pos = kHeaderSize;
    const HuffTable dcTable = HuffTable::deserialize(data, size, pos, kDcAlphabet);
    const HuffTable acTable = HuffTable::deserialize(data, size, pos, kAcAlphabet);

    QuantizedImage qi;
    qi.width = static_cast<int>(width);
    qi.height = static_cast<int>(height);
    qi.quality = quality;
    qi.blocksX = static_cast<int>((width + 7) / 8);
    qi.blocksY = static_cast<int>((height + 7) / 8);
    const size_t numBlocks = static_cast<size_t>(qi.blocksX) * qi.blocksY;
    qi.zigzag.assign(numBlocks * 64, 0);

    BitReader br(data + pos, size - pos);
    int prevDC = 0;
    for (size_t b = 0; b < numBlocks; ++b) {
        int16_t* blk = &qi.zigzag[b * 64];
        // DC: category, then amplitude bits, then add to the previous DC.
        const int dcSize = dcTable.decodeSymbol(br);
        const int diff = extendAmplitude(br.bits(dcSize), dcSize);
        prevDC += diff;
        blk[0] = static_cast<int16_t>(prevDC);
        // AC: (run, size) symbols until EOB or the block is full.
        int k = 1;
        while (k < 64) {
            const int sym = acTable.decodeSymbol(br);
            const int run = sym >> 4;
            const int sz = sym & 15;
            if (sz == 0) {
                if (run == 15) {  // ZRL: sixteen zeros
                    k += 16;
                    if (k > 64) throw std::runtime_error("decode: run overflows block");
                    continue;
                }
                if (sym != kEOB) throw std::runtime_error("decode: invalid AC symbol");
                break;  // EOB
            }
            k += run;
            if (k > 63) throw std::runtime_error("decode: run overflows block");
            blk[k++] = static_cast<int16_t>(extendAmplitude(br.bits(sz), sz));
        }
    }
    return qi;
}

Image decode(const uint8_t* data, size_t size, DecodeTimings* timings, bool fused) {
    const auto tStart = Clock::now();
    DecodeTimings local;
    DecodeTimings* tm = timings ? timings : &local;
    *tm = DecodeTimings{};

    // ---- Stage 1: header, tables, Huffman-decode all blocks.
    auto t0 = Clock::now();
    const QuantizedImage qi = decodeCoefficients(data, size);
    const uint32_t width = static_cast<uint32_t>(qi.width);
    const uint32_t height = static_cast<uint32_t>(qi.height);
    const int quality = qi.quality;
    const int blocksX = qi.blocksX, blocksY = qi.blocksY;
    const size_t numBlocks = static_cast<size_t>(blocksX) * blocksY;
    const std::vector<int16_t>& zigzag = qi.zigzag;
    tm->bitdecode_ms = msSince(t0);

    const QTable q = makeQuantTable(quality);
    const auto& zz = zigzagOrder();
    const int padW = blocksX * 8, padH = blocksY * 8;
    std::vector<uint8_t> padded(static_cast<size_t>(padW) * padH);

    // De-zigzag one block and multiply by the Q-table -> natural-order coefficients.
    auto dequantizeBlock = [&](const int16_t* in, double* out) {
        for (int i = 0; i < 64; ++i) {
            const int natural = zz[i];
            out[natural] = static_cast<double>(in[i]) * q[natural];
        }
    };
    // Inverse DCT of one block, undo the level shift, round, clamp to [0, 255], store.
    auto idctBlock = [&](const double* coef, int bx, int by) {
        double block[64];
        inverseDCT8x8(coef, block);
        for (int r = 0; r < 8; ++r) {
            uint8_t* row = &padded[static_cast<size_t>(by * 8 + r) * padW + bx * 8];
            for (int c = 0; c < 8; ++c) {
                const double v = roundHalfUp(block[r * 8 + c] + 128.0);
                row[c] = static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
            }
        }
    };

    if (fused) {
        t0 = Clock::now();
        for (int by = 0; by < blocksY; ++by) {
            for (int bx = 0; bx < blocksX; ++bx) {
                double coef[64];
                dequantizeBlock(&zigzag[(static_cast<size_t>(by) * blocksX + bx) * 64], coef);
                idctBlock(coef, bx, by);
            }
        }
        tm->idct_ms = msSince(t0);  // dequantization + IDCT together
    } else {
        // ---- Stage 2: de-zigzag and dequantize (multiply by the Q-table).
        t0 = Clock::now();
        std::vector<double> coeffs(numBlocks * 64);
        for (size_t b = 0; b < numBlocks; ++b) dequantizeBlock(&zigzag[b * 64], &coeffs[b * 64]);
        tm->dequant_ms = msSince(t0);

        // ---- Stage 3: inverse DCT, undo level shift, round, clamp to [0, 255].
        t0 = Clock::now();
        for (int by = 0; by < blocksY; ++by) {
            for (int bx = 0; bx < blocksX; ++bx) {
                idctBlock(&coeffs[(static_cast<size_t>(by) * blocksX + bx) * 64], bx, by);
            }
        }
        tm->idct_ms = msSince(t0);
    }

    // ---- Stage 4: crop the padding away.
    t0 = Clock::now();
    Image img(static_cast<int>(width), static_cast<int>(height));
    for (uint32_t y = 0; y < height; ++y) {
        std::memcpy(&img.pixels[static_cast<size_t>(y) * width], &padded[static_cast<size_t>(y) * padW], width);
    }
    tm->crop_ms = msSince(t0);
    tm->total_ms = msSince(tStart);
    return img;
}

}  // namespace jlc
