// jlc_bench - per-stage timing of the encoder and decoder.
//
//   jlc_bench [--sizes 512,1024,2048] [--quality 10,50,90] [--reps 5]
//             [--image file.pgm] [--csv out.csv] [--label name] [--fused 0|1]
//
// Without --image, deterministic synthetic pictures of the requested square sizes
// are used. Each configuration is run `reps` times (after one warm-up run) and the
// MEDIAN of every stage is reported, which is robust against a noisy node.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "jlc/codec.hpp"
#include "jlc/image.hpp"
#include "jlc/metrics.hpp"

using namespace jlc;

namespace {

std::vector<int> parseList(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) v.push_back(std::atoi(item.c_str()));
    }
    return v;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> sizes = {512, 1024, 2048};
    std::vector<int> qualities = {10, 50, 90};
    int reps = 5;
    std::string imagePath, csvPath, label = "synthetic";
    bool fused = false;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--sizes")) sizes = parseList(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--quality")) qualities = parseList(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--reps")) reps = std::max(1, std::atoi(argv[i + 1]));
        else if (!std::strcmp(argv[i], "--image")) imagePath = argv[i + 1];
        else if (!std::strcmp(argv[i], "--csv")) csvPath = argv[i + 1];
        else if (!std::strcmp(argv[i], "--label")) label = argv[i + 1];
        else if (!std::strcmp(argv[i], "--fused")) fused = std::atoi(argv[i + 1]) != 0;
        else {
            std::cerr << "unknown option " << argv[i] << "\n";
            return 2;
        }
    }

    std::vector<Image> images;
    std::vector<std::string> names;
    try {
        if (!imagePath.empty()) {
            images.push_back(readPGM(imagePath));
            names.push_back(imagePath);
        } else {
            for (int s : sizes) {
                images.push_back(makeSyntheticImage(s, s));
                names.push_back(label + "_" + std::to_string(s));
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    const char* header =
        "image,width,height,quality,bytes,bpp,psnr,"
        "enc_total_ms,enc_dct_ms,enc_quant_zigzag_ms,enc_scan_ms,enc_huff_build_ms,enc_bitpack_ms,"
        "dec_total_ms,dec_bitdecode_ms,dec_dequant_ms,dec_idct_ms,dec_crop_ms,enc_mpix_s,dec_mpix_s";
    std::ofstream csv;
    if (!csvPath.empty()) {
        csv.open(csvPath);
        csv << header << "\n";
    }
    std::printf("%s\n", header);

    for (size_t k = 0; k < images.size(); ++k) {
        const Image& img = images[k];
        for (int q : qualities) {
            std::vector<double> e[6], d[5];
            std::vector<uint8_t> bytes;
            for (int r = -1; r < reps; ++r) {  // r = -1 is the warm-up run, not recorded
                EncodeTimings et;
                bytes = encode(img, q, &et, fused);
                DecodeTimings dt;
                Image back = decode(bytes, &dt, fused);
                if (r < 0) continue;
                e[0].push_back(et.total_ms); e[1].push_back(et.dct_ms); e[2].push_back(et.quant_zigzag_ms);
                e[3].push_back(et.scan_ms);  e[4].push_back(et.huff_build_ms); e[5].push_back(et.bitpack_ms);
                d[0].push_back(dt.total_ms); d[1].push_back(dt.bitdecode_ms); d[2].push_back(dt.dequant_ms);
                d[3].push_back(dt.idct_ms);  d[4].push_back(dt.crop_ms);
            }
            const Image back = decode(bytes);
            const double mpix = img.numPixels() / 1e6;
            char line[1024];
            std::snprintf(line, sizeof line,
                          "%s,%d,%d,%d,%zu,%.5f,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f",
                          names[k].c_str(), img.width, img.height, q, bytes.size(),
                          8.0 * bytes.size() / img.numPixels(), psnr(img, back), median(e[0]), median(e[1]),
                          median(e[2]), median(e[3]), median(e[4]), median(e[5]), median(d[0]), median(d[1]),
                          median(d[2]), median(d[3]), median(d[4]), mpix / (median(e[0]) / 1e3),
                          mpix / (median(d[0]) / 1e3));
            std::printf("%s\n", line);
            if (csv.is_open()) csv << line << "\n";
        }
    }
    return 0;
}
