#!/usr/bin/env python3
"""Cut the code excerpts shown in docs/guide.tex out of the real source files.

Each excerpt = lines from the first line containing START up to (not including) the first
later line containing END. Doing this by script means the listings in the PDF can never drift
away from the code you actually run.
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "docs", "snippets")

EXCERPTS = [
    # name,            file,                        start marker,                                   end marker
    ("dct_matrix",     "src/dct.cpp",               "struct DctMatrix {",                          "const double* dctMatrix() {"),
    ("dct_forward",    "src/dct.cpp",               "void forwardDCT8x8(const double* in",          "void inverseDCT8x8(const double* in"),
    ("quant_table",    "src/quant.cpp",             "QTable makeQuantTable(int quality) {",         "}  // namespace jlc"),
    ("quant_round",    "include/jlc/quant.hpp",     "constexpr double kTieEpsilon",                 "}  // namespace jlc"),
    ("zigzag",         "src/zigzag.cpp",            "std::array<int, 64> buildZigzag() {",          "}  // namespace"),
    ("rle",            "src/rle.cpp",               "int category(int v) {",                        "}  // namespace jlc"),
    ("huff_merge",     "src/huffman.cpp",           "    // Min-heap ordered by",                     "    // Code length of a leaf"),
    ("huff_canonical", "src/huffman.cpp",           "    // Assign canonical codes",                 "void HuffTable::serialize"),
    ("huff_decode",    "src/huffman.cpp",           "int HuffTable::decodeSymbol",                  "}  // namespace jlc"),
    ("encode_main",    "src/codec.cpp",             "std::vector<uint8_t> encode(const Image& img", "// Decoder"),
    ("decode_blocks",  "src/codec.cpp",             "    BitReader br(data + pos, size - pos);",     "    return qi;"),
    ("api_example",    "include/jlc/codec.hpp",     "//   Image img = readPGM",                      "// File format"),
]


def extract(path, start, end):
    lines = open(os.path.join(ROOT, path)).read().split("\n")
    i = next(k for k, l in enumerate(lines) if start in l)
    j = next(k for k in range(i + 1, len(lines)) if end in lines[k])
    chunk = lines[i:j]
    while chunk and (not chunk[-1].strip() or chunk[-1].strip().startswith("// ----")):
        chunk.pop()
    return "\n".join(chunk) + "\n"


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, path, start, end in EXCERPTS:
        text = extract(path, start, end)
        with open(os.path.join(OUT, f"code_{name}.cpp"), "w") as f:
            f.write(text)
        print(f"code_{name}.cpp: {text.count(chr(10))} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
