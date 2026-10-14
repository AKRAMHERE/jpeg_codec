// image.hpp - minimal 8-bit grayscale image container and PGM (P5) file I/O.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace jlc {

// A grayscale image: one byte per pixel, rows stored top to bottom.
struct Image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;  // size = width * height

    Image() = default;
    Image(int w, int h) : width(w), height(h), pixels(static_cast<size_t>(w) * h, 0) {}

    uint8_t& at(int x, int y) { return pixels[static_cast<size_t>(y) * width + x]; }
    uint8_t at(int x, int y) const { return pixels[static_cast<size_t>(y) * width + x]; }
    size_t numPixels() const { return pixels.size(); }
};

// Reads a binary PGM ("P5", maxval 255). Throws std::runtime_error on failure.
Image readPGM(const std::string& path);

// Writes a binary PGM ("P5", maxval 255).
void writePGM(const std::string& path, const Image& img);

// Deterministic synthetic test picture (smooth gradients + sinusoids + hard edges
// + a little noise). Used by the benchmark so it needs no input files.
Image makeSyntheticImage(int width, int height, uint32_t seed = 1);

}  // namespace jlc
