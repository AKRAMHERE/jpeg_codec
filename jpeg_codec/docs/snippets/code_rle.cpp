int category(int v) {
    int a = std::abs(v);
    int n = 0;
    while (a != 0) {
        ++n;
        a >>= 1;
    }
    return n;
}

uint16_t amplitudeBits(int v, int size) {
    if (size == 0) return 0;
    const int mask = (1 << size) - 1;
    return static_cast<uint16_t>((v >= 0 ? v : v + mask) & mask);
}

int extendAmplitude(uint32_t bits, int size) {
    if (size == 0) return 0;
    // If the top bit is set the value is positive; otherwise it was stored as v + 2^size - 1.
    if (bits >= (1u << (size - 1))) return static_cast<int>(bits);
    return static_cast<int>(bits) - ((1 << size) - 1);
}

void tokenizeBlock(const int16_t* zz, int& prevDC, std::vector<Token>& out) {
    // --- DC: code the difference to the previous block (DPCM).
    const int diff = zz[0] - prevDC;
    prevDC = zz[0];
    const int dcSize = category(diff);
    out.push_back(Token{0, static_cast<uint8_t>(dcSize), static_cast<uint8_t>(dcSize),
                        amplitudeBits(diff, dcSize)});

    // --- AC: run-length code the zeros between non-zero coefficients.
    int run = 0;
    for (int k = 1; k < 64; ++k) {
        const int v = zz[k];
        if (v == 0) {
            ++run;
            continue;
        }
        while (run >= 16) {  // a run of 16 zeros is emitted as its own symbol
            out.push_back(Token{1, kZRL, 0, 0});
            run -= 16;
        }
        const int size = category(v);
        out.push_back(Token{1, static_cast<uint8_t>((run << 4) | size), static_cast<uint8_t>(size),
                            amplitudeBits(v, size)});
        run = 0;
    }
    if (run > 0) out.push_back(Token{1, kEOB, 0, 0});  // trailing zeros: end of block
}
