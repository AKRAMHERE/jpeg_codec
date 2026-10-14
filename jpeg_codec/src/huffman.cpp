#include "jlc/huffman.hpp"

#include <algorithm>
#include <queue>
#include <stdexcept>

namespace jlc {

namespace {

struct Node {
    uint64_t freq;
    int minSymbol;  // smallest symbol in this subtree (deterministic tie-break)
    int left;       // child indices, -1 for a leaf
    int right;
    int symbol;     // only meaningful for leaves
};

// One attempt at building a Huffman tree; returns code lengths (may exceed any limit).
std::vector<uint8_t> buildLengthsOnce(const std::vector<uint64_t>& freq) {
    std::vector<Node> nodes;
    for (size_t s = 0; s < freq.size(); ++s) {
        if (freq[s] > 0) nodes.push_back(Node{freq[s], static_cast<int>(s), -1, -1, static_cast<int>(s)});
    }
    std::vector<uint8_t> lengths(freq.size(), 0);
    if (nodes.empty()) return lengths;
    if (nodes.size() == 1) {
        lengths[nodes[0].symbol] = 1;
        return lengths;
    }

    // Min-heap ordered by (frequency, smallest symbol).
    auto later = [&nodes](int a, int b) {
        if (nodes[a].freq != nodes[b].freq) return nodes[a].freq > nodes[b].freq;
        return nodes[a].minSymbol > nodes[b].minSymbol;
    };
    std::priority_queue<int, std::vector<int>, decltype(later)> heap(later);
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i) heap.push(i);

    while (heap.size() > 1) {
        const int a = heap.top();
        heap.pop();
        const int b = heap.top();
        heap.pop();
        Node parent{nodes[a].freq + nodes[b].freq, std::min(nodes[a].minSymbol, nodes[b].minSymbol),
                    a, b, -1};
        nodes.push_back(parent);
        heap.push(static_cast<int>(nodes.size()) - 1);
    }

    // Code length of a leaf = its depth in the tree.
    std::vector<std::pair<int, int>> stack;  // (node index, depth)
    stack.push_back({heap.top(), 0});
    while (!stack.empty()) {
        auto [idx, depth] = stack.back();
        stack.pop_back();
        if (nodes[idx].left < 0) {
            lengths[nodes[idx].symbol] = static_cast<uint8_t>(depth > 255 ? 255 : depth);
        } else {
            stack.push_back({nodes[idx].left, depth + 1});
            stack.push_back({nodes[idx].right, depth + 1});
        }
    }
    return lengths;
}

}  // namespace

std::vector<uint8_t> huffmanCodeLengths(std::vector<uint64_t> freq, int maxLen) {
    for (;;) {
        std::vector<uint8_t> lengths = buildLengthsOnce(freq);
        const int longest = lengths.empty() ? 0 : *std::max_element(lengths.begin(), lengths.end());
        if (longest <= maxLen) return lengths;
        // Too deep: flatten the distribution and try again.
        for (auto& f : freq) {
            if (f > 0) f = (f + 1) / 2;
        }
    }
}

HuffTable::HuffTable(const std::vector<uint8_t>& lengths) : length_(lengths) {
    for (uint8_t l : length_) {
        if (l > kMaxCodeLength) throw std::runtime_error("huffman: code length too large");
    }
    finalize();
}

void HuffTable::finalize() {
    const int n = static_cast<int>(length_.size());
    code_.assign(n, 0);
    count_.fill(0);
    first_.fill(0);
    offset_.fill(0);
    sorted_.clear();
    maxLength_ = 0;

    // Symbols in canonical order: by (length, symbol).
    std::vector<int> order;
    for (int s = 0; s < n; ++s) {
        if (length_[s] > 0) {
            order.push_back(s);
            ++count_[length_[s]];
            maxLength_ = std::max<int>(maxLength_, length_[s]);
        }
    }
    std::stable_sort(order.begin(), order.end(), [this](int a, int b) { return length_[a] < length_[b]; });
    for (int s : order) sorted_.push_back(static_cast<uint8_t>(s));

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

void HuffTable::serialize(std::vector<uint8_t>& out) const {
    for (int len = 1; len <= kMaxCodeLength; ++len) {
        if (count_[len] > 255) throw std::runtime_error("huffman: too many codes of one length");
        out.push_back(static_cast<uint8_t>(count_[len]));
    }
    out.insert(out.end(), sorted_.begin(), sorted_.end());
}

HuffTable HuffTable::deserialize(const uint8_t* data, size_t size, size_t& pos, int alphabet) {
    if (pos + kMaxCodeLength > size) throw std::runtime_error("huffman: truncated table");
    int counts[kMaxCodeLength + 1] = {0};
    int total = 0;
    for (int len = 1; len <= kMaxCodeLength; ++len) {
        counts[len] = data[pos++];
        total += counts[len];
    }
    if (total > alphabet || pos + total > size) throw std::runtime_error("huffman: corrupt table");
    std::vector<uint8_t> lengths(alphabet, 0);
    for (int len = 1; len <= kMaxCodeLength; ++len) {
        for (int i = 0; i < counts[len]; ++i) {
            const uint8_t sym = data[pos++];
            if (sym >= alphabet || lengths[sym] != 0) throw std::runtime_error("huffman: corrupt table");
            lengths[sym] = static_cast<uint8_t>(len);
        }
    }
    return HuffTable(lengths);
}

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

}  // namespace jlc
