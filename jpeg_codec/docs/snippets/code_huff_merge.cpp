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
