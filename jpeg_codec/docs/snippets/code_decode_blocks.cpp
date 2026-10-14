    BitReader br(data + pos, size - pos);
    int prevDC = 0;
    for (size_t b = 0; b < numBlocks; ++b) {
        int16_t* blk = &qi.zigzag[b * 64];
        // DC: category, then amplitude bits, then add to the previous DC.
        const int dcSize = dcTable.decodeSymbol(br);
        const int diff = extendAmplitude(br.bits(dcSize), dcSize);
        prevDC += diff;
        blk[0] = static_cast<int16_t>(prevDC);
        // AC: (run, size) symbols until EOB or the block is full.
        int k = 1;
        while (k < 64) {
            const int sym = acTable.decodeSymbol(br);
            const int run = sym >> 4;
            const int sz = sym & 15;
            if (sz == 0) {
                if (run == 15) {  // ZRL: sixteen zeros
                    k += 16;
                    if (k > 64) throw std::runtime_error("decode: run overflows block");
                    continue;
                }
                if (sym != kEOB) throw std::runtime_error("decode: invalid AC symbol");
                break;  // EOB
            }
            k += run;
            if (k > 63) throw std::runtime_error("decode: run overflows block");
            blk[k++] = static_cast<int16_t>(extendAmplitude(br.bits(sz), sz));
        }
    }
