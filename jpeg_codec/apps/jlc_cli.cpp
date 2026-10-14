// jlc - command line front end for the JPEG-like codec.
//
//   jlc encode in.pgm out.jlc [-q 50]
//   jlc decode in.jlc out.pgm
//   jlc eval   in.pgm [-q 50]               round trip + metrics on one line
//   jlc sweep  in.pgm [-Q 1,5,10,...]       CSV rate-distortion table
//   jlc info   in.jlc                       print header fields
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "jlc/codec.hpp"
#include "jlc/image.hpp"
#include "jlc/metrics.hpp"

using namespace jlc;

namespace {

std::vector<uint8_t> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void writeFile(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

void usage() {
    std::cerr << "usage:\n"
                 "  jlc encode in.pgm out.jlc [-q QUALITY]\n"
                 "  jlc decode in.jlc out.pgm\n"
                 "  jlc eval   in.pgm [-q QUALITY]\n"
                 "  jlc sweep  in.pgm [-Q q1,q2,...]\n"
                 "  jlc info   in.jlc\n";
}

int parseQuality(int argc, char** argv, int firstOpt, int fallback) {
    for (int i = firstOpt; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "-q") == 0) return std::atoi(argv[i + 1]);
    }
    return fallback;
}

std::vector<int> parseList(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) v.push_back(std::atoi(item.c_str()));
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        usage();
        return 2;
    }
    const std::string cmd = argv[1];
    try {
        if (cmd == "encode" && argc >= 4) {
            const Image img = readPGM(argv[2]);
            const int q = parseQuality(argc, argv, 4, 50);
            const auto bytes = encode(img, q);
            writeFile(argv[3], bytes);
            std::printf("%s: %dx%d, quality %d -> %zu bytes (%.3f bpp, ratio %.2f:1)\n", argv[3], img.width,
                        img.height, q, bytes.size(), 8.0 * bytes.size() / img.numPixels(),
                        static_cast<double>(img.numPixels()) / bytes.size());
        } else if (cmd == "decode" && argc >= 4) {
            const auto bytes = readFile(argv[2]);
            writePGM(argv[3], decode(bytes));
        } else if (cmd == "eval") {
            const Image img = readPGM(argv[2]);
            const int q = parseQuality(argc, argv, 3, 50);
            const auto bytes = encode(img, q);
            const Image back = decode(bytes);
            const double m = mse(img, back);
            std::printf("quality=%d bytes=%zu bpp=%.4f ratio=%.3f mse=%.4f psnr=%.3f maxerr=%d\n", q, bytes.size(),
                        8.0 * bytes.size() / img.numPixels(), static_cast<double>(img.numPixels()) / bytes.size(),
                        m, psnrFromMse(m), maxAbsError(img, back));
        } else if (cmd == "sweep") {
            const Image img = readPGM(argv[2]);
            std::vector<int> qs = {1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100};
            for (int i = 3; i + 1 < argc; ++i) {
                if (std::strcmp(argv[i], "-Q") == 0) qs = parseList(argv[i + 1]);
            }
            std::printf("quality,bytes,bpp,ratio,mse,psnr,maxerr\n");
            for (int q : qs) {
                const auto bytes = encode(img, q);
                const Image back = decode(bytes);
                const double m = mse(img, back);
                std::printf("%d,%zu,%.6f,%.6f,%.6f,%.6f,%d\n", q, bytes.size(), 8.0 * bytes.size() / img.numPixels(),
                            static_cast<double>(img.numPixels()) / bytes.size(), m, psnrFromMse(m),
                            maxAbsError(img, back));
            }
        } else if (cmd == "info") {
            const auto b = readFile(argv[2]);
            if (b.size() < 13 || std::memcmp(b.data(), "JLC1", 4) != 0) throw std::runtime_error("not a JLC1 file");
            auto u32 = [&b](size_t o) {
                return static_cast<uint32_t>(b[o]) | (static_cast<uint32_t>(b[o + 1]) << 8) |
                       (static_cast<uint32_t>(b[o + 2]) << 16) | (static_cast<uint32_t>(b[o + 3]) << 24);
            };
            std::printf("JLC1 stream: %ux%u, quality %d, %zu bytes\n", u32(4), u32(8), b[12], b.size());
        } else {
            usage();
            return 2;
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
