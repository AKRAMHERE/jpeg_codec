int HuffTable::decodeSymbol(BitReader& br) const {
    uint64_t code = 0;
    for (int len = 1; len <= maxLength_; ++len) {
        code = (code << 1) | static_cast<uint64_t>(br.bit());
        if (count_[len] > 0 && code >= first_[len] && code < first_[len] + static_cast<uint64_t>(count_[len])) {
            return sorted_[offset_[len] + static_cast<int>(code - first_[len])];
        }
    }
    throw std::runtime_error("huffman: invalid code in bitstream");
}
