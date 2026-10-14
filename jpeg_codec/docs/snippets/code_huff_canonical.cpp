    // Assign canonical codes: code(next) = (code(prev) + 1) << (len(next) - len(prev)).
    uint64_t code = 0;
    int prevLen = 0;
    int index = 0;
    for (int len = 1; len <= maxLength_; ++len) {
        code <<= (len - prevLen);
        prevLen = len;
        first_[len] = code;
        offset_[len] = index;
        for (int i = 0; i < count_[len]; ++i) {
            code_[sorted_[index]] = static_cast<uint32_t>(code);
            ++code;
            ++index;
        }
    }
}
