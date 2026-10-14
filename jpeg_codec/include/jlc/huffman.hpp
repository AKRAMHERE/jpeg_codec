// huffman.hpp - Huffman code construction, canonical codes, table (de)serialization,
//               and bit-by-bit canonical decoding.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "jlc/bitio.hpp"

namespace jlc {

constexpr int kMaxCodeLength = 32;

// Computes Huffman code lengths for symbols 0..freq.size()-1 (length 0 = symbol unused).
//  * Deterministic tie-breaking: the two nodes with the smallest (frequency, smallest
//    symbol in the subtree) are merged first. The Python reference does the same, so
//    both implementations produce byte-identical files.
//  * If the longest code would exceed `maxLen`, frequencies are halved (rounded up)
//    and the tree rebuilt until it fits.
//  * A single used symbol gets length 1.
std::vector<uint8_t> huffmanCodeLengths(std::vector<uint64_t> freq, int maxLen = kMaxCodeLength);

// A canonical Huffman code: codes are assigned in order of (length, symbol), so the
// decoder only needs the lengths.
class HuffTable {
public:
    HuffTable() = default;
    // Build from code lengths (one per symbol; 0 = unused).
    explicit HuffTable(const std::vector<uint8_t>& lengths);

    int alphabetSize() const { return static_cast<int>(length_.size()); }
    uint8_t length(int symbol) const { return length_[symbol]; }
    uint32_t code(int symbol) const { return code_[symbol]; }

    // Serialized form (JPEG-like): 32 bytes "number of codes of length 1..32"
    // followed by the used symbols in canonical order.
    void serialize(std::vector<uint8_t>& out) const;
    static HuffTable deserialize(const uint8_t* data, size_t size, size_t& pos, int alphabet);

    // Read one symbol from the bitstream (canonical, bit by bit).
    int decodeSymbol(BitReader& br) const;

private:
    void finalize();
    std::vector<uint8_t> length_;
    std::vector<uint32_t> code_;
    // Canonical decoding tables, indexed by code length 1..32.
    std::array<int, kMaxCodeLength + 1> count_{};         // number of codes of that length
    std::array<uint64_t, kMaxCodeLength + 1> first_{};    // first code of that length
    std::array<int, kMaxCodeLength + 1> offset_{};        // index into sorted_ of that first code
    std::vector<uint8_t> sorted_;                         // symbols in canonical order
    int maxLength_ = 0;
};

}  // namespace jlc
