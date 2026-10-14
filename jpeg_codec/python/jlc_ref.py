#!/usr/bin/env python3
"""NumPy reference model of the JLC1 codec.

This is an independent re-implementation of the C++ library. It follows exactly the same
algorithm and file format, so the two can be cross-checked:

    python3 jlc_ref.py encode in.pgm out.jlc -q 50
    python3 jlc_ref.py decode in.jlc out.pgm

The transform stages (DCT, quantization, zig-zag) are vectorised with NumPy over all
blocks at once. The entropy stages (run-length tokens, Huffman, bit packing) are plain
Python loops written for clarity, not speed - use small images (<= 512x512).
"""
import argparse
import heapq
import struct
import sys

import numpy as np

from make_test_images import read_pgm, write_pgm

TIE_EPS = 1e-9  # same deterministic tie-break as the C++ code (see quant.hpp)


def round_half_up(x):
    return np.floor(x + 0.5 + TIE_EPS)


MAGIC = b"JLC1"
DC_ALPHABET, AC_ALPHABET = 16, 256
EOB, ZRL = 0x00, 0xF0
MAX_CODE_LEN = 32

BASE_Q = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99], dtype=np.int64)


# --------------------------------------------------------------------------- transform
def dct_matrix():
    """C[u, x] = alpha(u) cos((2x+1) u pi / 16); orthonormal, so the inverse is C^T."""
    c = np.zeros((8, 8))
    for u in range(8):
        alpha = np.sqrt(1 / 8) if u == 0 else np.sqrt(2 / 8)
        for x in range(8):
            c[u, x] = alpha * np.cos((2 * x + 1) * u * np.pi / 16)
    return c


C = dct_matrix()


def zigzag_order():
    order = []
    for s in range(15):
        lo, hi = max(0, s - 7), min(s, 7)
        rows = range(lo, hi + 1) if s % 2 == 1 else range(hi, lo - 1, -1)
        order += [r * 8 + (s - r) for r in rows]
    return np.array(order)


ZZ = zigzag_order()


def quant_table(quality):
    quality = int(min(max(quality, 1), 100))
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    return np.clip((BASE_Q * scale + 50) // 100, 1, 255)


def pad_to_blocks(img):
    h, w = img.shape
    ph, pw = (-h) % 8, (-w) % 8
    return np.pad(img, ((0, ph), (0, pw)), mode="edge")


def to_blocks(a):
    h, w = a.shape
    return a.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)


def from_blocks(b, h, w):
    return b.reshape(h // 8, w // 8, 8, 8).transpose(0, 2, 1, 3).reshape(h, w)


def forward_dct_blocks(img):
    """Return DCT coefficients of every 8x8 block, shape (nblocks, 8, 8)."""
    blocks = to_blocks(pad_to_blocks(img).astype(np.float64) - 128.0)
    return C @ blocks @ C.T


def analyze(img, quality):
    """DCT + quantization + zig-zag -> int array (nblocks, 64)."""
    coeffs = forward_dct_blocks(img).reshape(-1, 64)
    q = quant_table(quality)
    return round_half_up(coeffs / q).astype(np.int64)[:, ZZ]


def reconstruct(zigzag, quality, h, w):
    """Dequantize + inverse DCT + level shift + clamp, from (nblocks, 64) zig-zag coefficients."""
    ph, pw = h + (-h) % 8, w + (-w) % 8
    nat = np.zeros_like(zigzag, dtype=np.float64)
    nat[:, ZZ] = zigzag * quant_table(quality)[ZZ]
    blocks = C.T @ nat.reshape(-1, 8, 8) @ C
    img = round_half_up(from_blocks(blocks, ph, pw) + 128.0)
    return np.clip(img, 0, 255).astype(np.uint8)[:h, :w]


# --------------------------------------------------------------------------- run-length
def category(v):
    return abs(int(v)).bit_length()


def amplitude_bits(v, size):
    return 0 if size == 0 else (v if v >= 0 else v + (1 << size) - 1) & ((1 << size) - 1)


def extend(bits, size):
    if size == 0:
        return 0
    return bits if bits >= (1 << (size - 1)) else bits - ((1 << size) - 1)


def tokenize(zigzag):
    """List of (is_ac, symbol, nbits, amp) for all blocks."""
    tokens, prev_dc = [], 0
    for blk in zigzag:
        diff = int(blk[0]) - prev_dc
        prev_dc = int(blk[0])
        s = category(diff)
        tokens.append((0, s, s, amplitude_bits(diff, s)))
        run = 0
        for k in range(1, 64):
            v = int(blk[k])
            if v == 0:
                run += 1
                continue
            while run >= 16:
                tokens.append((1, ZRL, 0, 0))
                run -= 16
            s = category(v)
            tokens.append((1, (run << 4) | s, s, amplitude_bits(v, s)))
            run = 0
        if run > 0:
            tokens.append((1, EOB, 0, 0))
    return tokens


# --------------------------------------------------------------------------- huffman
def _lengths_once(freq):
    heap = [(f, s, ("leaf", s)) for s, f in enumerate(freq) if f > 0]
    lengths = [0] * len(freq)
    if not heap:
        return lengths
    if len(heap) == 1:
        lengths[heap[0][1]] = 1
        return lengths
    heapq.heapify(heap)  # ordered by (freq, smallest symbol): unique, so payload never compared
    while len(heap) > 1:
        fa, ma, na = heapq.heappop(heap)
        fb, mb, nb = heapq.heappop(heap)
        heapq.heappush(heap, (fa + fb, min(ma, mb), ("node", na, nb)))
    stack = [(heap[0][2], 0)]
    while stack:
        node, depth = stack.pop()
        if node[0] == "leaf":
            lengths[node[1]] = depth
        else:
            stack.append((node[1], depth + 1))
            stack.append((node[2], depth + 1))
    return lengths


def huffman_lengths(freq, max_len=MAX_CODE_LEN):
    freq = list(freq)
    while True:
        lengths = _lengths_once(freq)
        if max(lengths) <= max_len:
            return lengths
        freq = [(f + 1) // 2 if f > 0 else 0 for f in freq]


class HuffTable:
    """Canonical Huffman table built from code lengths."""

    def __init__(self, lengths):
        self.length = list(lengths)
        order = sorted((l, s) for s, l in enumerate(self.length) if l > 0)
        self.sorted_syms = [s for _, s in order]
        self.code = [0] * len(self.length)
        self.count = [0] * (MAX_CODE_LEN + 1)
        self.first = [0] * (MAX_CODE_LEN + 1)
        self.offset = [0] * (MAX_CODE_LEN + 1)
        for l, _ in order:
            self.count[l] += 1
        self.max_len = max([l for l, _ in order], default=0)
        code, prev_len, idx = 0, 0, 0
        for l in range(1, self.max_len + 1):
            code <<= (l - prev_len)
            prev_len = l
            self.first[l], self.offset[l] = code, idx
            for _ in range(self.count[l]):
                self.code[self.sorted_syms[idx]] = code
                code += 1
                idx += 1

    def serialize(self):
        return bytes(self.count[1:MAX_CODE_LEN + 1]) + bytes(self.sorted_syms)

    @staticmethod
    def deserialize(data, pos, alphabet):
        counts = list(data[pos:pos + MAX_CODE_LEN])
        pos += MAX_CODE_LEN
        total = sum(counts)
        syms = list(data[pos:pos + total])
        pos += total
        lengths, i = [0] * alphabet, 0
        for l, n in enumerate(counts, start=1):
            for _ in range(n):
                lengths[syms[i]] = l
                i += 1
        return HuffTable(lengths), pos

    def decode_symbol(self, reader):
        code = 0
        for l in range(1, self.max_len + 1):
            code = (code << 1) | reader.bit()
            if self.count[l] and self.first[l] <= code < self.first[l] + self.count[l]:
                return self.sorted_syms[self.offset[l] + code - self.first[l]]
        raise ValueError("invalid Huffman code")


class BitWriter:
    def __init__(self):
        self.out, self.acc, self.n = bytearray(), 0, 0

    def put(self, value, nbits):
        if nbits == 0:
            return
        self.acc = (self.acc << nbits) | (value & ((1 << nbits) - 1))
        self.n += nbits
        while self.n >= 8:
            self.out.append((self.acc >> (self.n - 8)) & 0xFF)
            self.n -= 8
        self.acc &= (1 << self.n) - 1

    def flush(self):
        if self.n:
            self.put(0, 8 - self.n)


class BitReader:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def bit(self):
        byte, off = divmod(self.pos, 8)
        if byte >= len(self.data):
            raise ValueError("unexpected end of data")
        self.pos += 1
        return (self.data[byte] >> (7 - off)) & 1

    def bits(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | self.bit()
        return v


# --------------------------------------------------------------------------- codec
def encode(img, quality):
    h, w = img.shape
    quality = int(min(max(quality, 1), 100))
    zz = analyze(img, quality)
    tokens = tokenize(zz)
    dc_freq, ac_freq = [0] * DC_ALPHABET, [0] * AC_ALPHABET
    for is_ac, sym, _, _ in tokens:
        (ac_freq if is_ac else dc_freq)[sym] += 1
    dc_t, ac_t = HuffTable(huffman_lengths(dc_freq)), HuffTable(huffman_lengths(ac_freq))
    out = bytearray(MAGIC + struct.pack("<IIB", w, h, quality))
    out += dc_t.serialize() + ac_t.serialize()
    bw = BitWriter()
    for is_ac, sym, nbits, amp in tokens:
        t = ac_t if is_ac else dc_t
        bw.put(t.code[sym], t.length[sym])
        bw.put(amp, nbits)
    bw.flush()
    return bytes(out + bw.out)


def decode_coefficients(data):
    if len(data) < 13 or data[:4] != MAGIC:
        raise ValueError("not a JLC1 stream")
    w, h, quality = struct.unpack("<IIB", data[4:13])
    pos = 13
    dc_t, pos = HuffTable.deserialize(data, pos, DC_ALPHABET)
    ac_t, pos = HuffTable.deserialize(data, pos, AC_ALPHABET)
    nblocks = ((h + 7) // 8) * ((w + 7) // 8)
    br = BitReader(data[pos:])
    zz = np.zeros((nblocks, 64), dtype=np.int64)
    prev_dc = 0
    for b in range(nblocks):
        size = dc_t.decode_symbol(br)
        prev_dc += extend(br.bits(size), size)
        zz[b, 0] = prev_dc
        k = 1
        while k < 64:
            sym = ac_t.decode_symbol(br)
            run, size = sym >> 4, sym & 15
            if size == 0:
                if run == 15:
                    k += 16
                    continue
                break
            k += run
            zz[b, k] = extend(br.bits(size), size)
            k += 1
    return zz, quality, h, w


def decode(data):
    zz, quality, h, w = decode_coefficients(data)
    return reconstruct(zz, quality, h, w)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("src"), e.add_argument("dst"), e.add_argument("-q", type=int, default=50)
    d = sub.add_parser("decode")
    d.add_argument("src"), d.add_argument("dst")
    args = ap.parse_args()
    if args.cmd == "encode":
        data = encode(read_pgm(args.src), args.q)
        open(args.dst, "wb").write(data)
        print(f"{args.dst}: {len(data)} bytes")
    else:
        write_pgm(args.dst, decode(open(args.src, "rb").read()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
