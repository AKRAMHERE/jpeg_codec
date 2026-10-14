// bitio.hpp - MSB-first bit writer / reader on top of a byte vector.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace jlc {

class BitWriter {
public:
    explicit BitWriter(std::vector<uint8_t>& out) : out_(out) {}

    // Append the lowest `n` bits of `value` (0 <= n <= 32), most significant bit first.
    void put(uint32_t value, int n) {
        if (n == 0) return;
        acc_ = (acc_ << n) | (value & mask(n));
        nbits_ += n;
        while (nbits_ >= 8) {
            out_.push_back(static_cast<uint8_t>(acc_ >> (nbits_ - 8)));
            nbits_ -= 8;
        }
        acc_ &= mask(nbits_);
    }

    // Pad the last partial byte with zero bits.
    void flush() {
        if (nbits_ > 0) put(0, 8 - nbits_);
    }

private:
    static uint64_t mask(int n) { return (n >= 64) ? ~0ull : ((1ull << n) - 1ull); }
    std::vector<uint8_t>& out_;
    uint64_t acc_ = 0;  // holds the not-yet-written bits (always fewer than 8 after put)
    int nbits_ = 0;
};

class BitReader {
public:
    BitReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    int bit() {
        if (bytePos_ >= size_) throw std::runtime_error("bitstream: unexpected end of data");
        int b = (data_[bytePos_] >> (7 - bitPos_)) & 1;
        if (++bitPos_ == 8) {
            bitPos_ = 0;
            ++bytePos_;
        }
        return b;
    }

    uint32_t bits(int n) {
        uint32_t v = 0;
        for (int i = 0; i < n; ++i) v = (v << 1) | static_cast<uint32_t>(bit());
        return v;
    }

private:
    const uint8_t* data_;
    size_t size_;
    size_t bytePos_ = 0;
    int bitPos_ = 0;
};

}  // namespace jlc
