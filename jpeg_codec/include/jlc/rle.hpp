// rle.hpp - turns quantized, zig-zag-ordered blocks into (symbol, extra-bits) tokens
//           and back. This is the "DPCM + run-length" stage of JPEG.
//
// For each block:
//   * DC coefficient: difference to the previous block's DC, coded as
//       symbol = category(diff)   (0..15)   + `category` extra bits (the amplitude)
//   * AC coefficients 1..63: zeros are counted, every non-zero value becomes
//       symbol = (run << 4) | category(value)   + `category` extra bits
//     with two special symbols:
//       0xF0 (ZRL) = "16 zeros, keep going"      0x00 (EOB) = "all remaining are zero"
#pragma once

#include <cstdint>
#include <vector>

namespace jlc {

constexpr int kDcAlphabet = 16;    // DC symbols: categories 0..15
constexpr int kAcAlphabet = 256;   // AC symbols: (run<<4)|size, only 242 combos are valid
constexpr uint8_t kEOB = 0x00;
constexpr uint8_t kZRL = 0xF0;

// One entropy-coding token: a Huffman symbol plus `nbits` raw amplitude bits.
struct Token {
    uint8_t isAC;    // 0 = DC table, 1 = AC table
    uint8_t symbol;  // symbol to Huffman-code
    uint8_t nbits;   // number of amplitude bits that follow the Huffman code
    uint16_t amp;    // the amplitude bits (low `nbits` bits)
};

// Number of bits needed to represent |v| (0 for v == 0). This is the JPEG "category".
int category(int v);

// Amplitude bits for value v in category `size`: positive values as themselves,
// negative values in one's-complement style (v + 2^size - 1).
uint16_t amplitudeBits(int v, int size);

// Inverse of amplitudeBits: recover the signed value from `size` raw bits.
int extendAmplitude(uint32_t bits, int size);

// Tokenize one block given in zig-zag order (64 values). `prevDC` is updated.
// Appends tokens to `out`.
void tokenizeBlock(const int16_t* zz, int& prevDC, std::vector<Token>& out);

}  // namespace jlc
