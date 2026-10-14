std::vector<uint8_t> encode(const Image& img, int quality, EncodeTimings* timings, bool fused) {
    const auto tStart = Clock::now();
    EncodeTimings local;
    EncodeTimings* tm = timings ? timings : &local;
    *tm = EncodeTimings{};

    QuantizedImage qi = analyze(img, quality, tm, fused);
    const size_t numBlocks = static_cast<size_t>(qi.blocksX) * qi.blocksY;

    // ---- Stage 3: DC prediction + run-length tokenization, count symbol frequencies.
    auto t0 = Clock::now();
    std::vector<Token> tokens;
    tokens.reserve(numBlocks * 8);
    int prevDC = 0;
    for (size_t b = 0; b < numBlocks; ++b) tokenizeBlock(&qi.zigzag[b * 64], prevDC, tokens);
    std::vector<uint64_t> dcFreq(kDcAlphabet, 0), acFreq(kAcAlphabet, 0);
    for (const Token& t : tokens) (t.isAC ? acFreq : dcFreq)[t.symbol]++;
    tm->scan_ms = msSince(t0);

    // ---- Stage 4: build the two Huffman tables from the symbol statistics.
    t0 = Clock::now();
    const HuffTable dcTable(huffmanCodeLengths(dcFreq));
    const HuffTable acTable(huffmanCodeLengths(acFreq));
    tm->huff_build_ms = msSince(t0);

    // ---- Stage 5: serialize header, tables and the entropy-coded payload.
    t0 = Clock::now();
    std::vector<uint8_t> out;
    out.reserve(kHeaderSize + 128 + tokens.size());
    out.insert(out.end(), kMagic, kMagic + 4);
    putU32(out, static_cast<uint32_t>(img.width));
    putU32(out, static_cast<uint32_t>(img.height));
    out.push_back(static_cast<uint8_t>(qi.quality));
    dcTable.serialize(out);
    acTable.serialize(out);
    {
        BitWriter bw(out);
        for (const Token& t : tokens) {
            const HuffTable& table = t.isAC ? acTable : dcTable;
            bw.put(table.code(t.symbol), table.length(t.symbol));
            bw.put(t.amp, t.nbits);
        }
        bw.flush();
    }
    tm->bitpack_ms = msSince(t0);
    tm->total_ms = msSince(tStart);
    return out;
}
