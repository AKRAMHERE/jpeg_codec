#include "jlc/image.hpp"

#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace jlc {

namespace {

// Reads the next whitespace-delimited token of a PGM header, skipping '#' comments.
std::string nextToken(const std::vector<char>& buf, size_t& pos) {
    while (pos < buf.size()) {
        char c = buf[pos];
        if (c == '#') {
            while (pos < buf.size() && buf[pos] != '\n') ++pos;
        } else if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            ++pos;
        } else {
            break;
        }
    }
    std::string tok;
    while (pos < buf.size()) {
        char c = buf[pos];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '#') break;
        tok.push_back(c);
        ++pos;
    }
    return tok;
}

}  // namespace

Image readPGM(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    size_t pos = 0;
    if (nextToken(buf, pos) != "P5") throw std::runtime_error(path + ": not a binary PGM (P5)");
    int w = 0, h = 0, maxval = 0;
    try {
        w = std::stoi(nextToken(buf, pos));
        h = std::stoi(nextToken(buf, pos));
        maxval = std::stoi(nextToken(buf, pos));
    } catch (const std::exception&) {
        throw std::runtime_error(path + ": malformed PGM header");
    }
    if (w <= 0 || h <= 0 || maxval != 255) {
        throw std::runtime_error(path + ": only 8-bit PGM (maxval 255) is supported");
    }
    ++pos;  // exactly one whitespace byte separates header and raster
    const size_t n = static_cast<size_t>(w) * h;
    if (buf.size() < pos + n) throw std::runtime_error(path + ": truncated pixel data");

    Image img(w, h);
    for (size_t i = 0; i < n; ++i) img.pixels[i] = static_cast<uint8_t>(buf[pos + i]);
    return img;
}

void writePGM(const std::string& path, const Image& img) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    f << "P5\n" << img.width << " " << img.height << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.pixels.data()),
            static_cast<std::streamsize>(img.pixels.size()));
    if (!f) throw std::runtime_error("write failed: " + path);
}

Image makeSyntheticImage(int width, int height, uint32_t seed) {
    Image img(width, height);
    uint32_t state = seed * 2654435761u + 12345u;
    auto rnd = [&state]() {  // tiny LCG, deterministic everywhere
        state = state * 1664525u + 1013904223u;
        return static_cast<int>((state >> 24) & 0xFF);
    };
    const double cx = 0.5 * width, cy = 0.5 * height;
    const double radius = 0.18 * (width < height ? width : height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double v = 110.0 + 40.0 * x / width + 30.0 * y / height;     // slow gradient
            v += 25.0 * std::sin(0.045 * x) * std::cos(0.031 * y);       // smooth texture
            v += 12.0 * std::sin(0.31 * (x + 0.7 * y));                  // finer texture
            if (((x / 64) + (y / 64)) % 2 == 0) v += 22.0;               // hard-edged tiles
            const double dx = x - cx, dy = y - cy;
            if (dx * dx + dy * dy < radius * radius) v = 215.0 - 0.1 * (x % 16);  // bright disc
            v += (rnd() - 128) / 40.0;                                   // sensor-like noise
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            img.at(x, y) = static_cast<uint8_t>(v + 0.5);
        }
    }
    return img;
}

}  // namespace jlc
