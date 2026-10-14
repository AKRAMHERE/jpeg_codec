// codec.hpp - public API of the JPEG-like grayscale image codec.
//
//   Image img = readPGM("photo.pgm");
//   std::vector<uint8_t> bytes = jlc::encode(img, /*quality=*/50);
//   Image back = jlc::decode(bytes);
//
// File format ("JLC1", all integers little endian):
//   offset 0   4 bytes  magic "JLC1"
//   offset 4   u32      width
//   offset 8   u32      height
//   offset 12  u8       quality (1..100)
//   then       DC Huffman table (32 count bytes + symbols)
//   then       AC Huffman table (32 count bytes + symbols)
//   then       entropy-coded payload (blocks in raster order, MSB-first bits,
//              zero-padded to a whole byte)
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "jlc/image.hpp"

namespace jlc {

// Wall-clock time (milliseconds) spent in each stage of encode().
struct EncodeTimings {
    double dct_ms = 0;           // pad + level shift + forward 8x8 DCT of every block
    double quant_zigzag_ms = 0;  // divide by Q-table, round, reorder into zig-zag order
    double scan_ms = 0;          // DC prediction + run-length tokenization + symbol counting
    double huff_build_ms = 0;    // build both Huffman tables
    double bitpack_ms = 0;       // write header, tables and the entropy-coded payload
    double total_ms = 0;
};

// Wall-clock time (milliseconds) spent in each stage of decode().
struct DecodeTimings {
    double bitdecode_ms = 0;  // parse header/tables, Huffman-decode every block
    double dequant_ms = 0;    // de-zigzag and multiply by Q-table
    double idct_ms = 0;       // inverse 8x8 DCT, level shift, round, clamp
    double crop_ms = 0;       // remove padding
    double total_ms = 0;
};

// Quantized coefficients of a whole image, 64 per block, each block in zig-zag order.
struct QuantizedImage {
    int width = 0, height = 0;          // original size
    int blocksX = 0, blocksY = 0;       // size in 8x8 blocks (after padding)
    int quality = 0;
    std::vector<int16_t> zigzag;        // blocksX * blocksY * 64 values
};

// Two execution strategies with IDENTICAL output (the tests verify this bit for bit):
//   fused = false  "staged": each stage runs over the whole image and stores a full-size
//                  intermediate buffer (8 bytes per pixel). Easy to profile stage by stage.
//   fused = true   DCT + quantization (encode) / dequantization + IDCT (decode) run per 8x8
//                  block in a small local buffer. No full-size double buffer -> much less
//                  memory traffic. Timings then report the merged stage in dct_ms / idct_ms
//                  and 0 for the stage that was folded in.

// Runs only DCT + quantization + zig-zag (no entropy coding). Useful for analysis/tests.
QuantizedImage analyze(const Image& img, int quality, EncodeTimings* timings = nullptr, bool fused = false);

// Compress an image. quality is clamped to [1, 100].
std::vector<uint8_t> encode(const Image& img, int quality, EncodeTimings* timings = nullptr, bool fused = false);

// Entropy-decodes a stream back to its quantized coefficients (no dequantization or
// inverse DCT). Lets the tests check that the lossless part of the codec is exactly lossless.
QuantizedImage decodeCoefficients(const uint8_t* data, size_t size);

// Decompress. Throws std::runtime_error on malformed or truncated input.
Image decode(const uint8_t* data, size_t size, DecodeTimings* timings = nullptr, bool fused = false);
inline Image decode(const std::vector<uint8_t>& bytes, DecodeTimings* timings = nullptr, bool fused = false) {
    return decode(bytes.data(), bytes.size(), timings, fused);
}

}  // namespace jlc
