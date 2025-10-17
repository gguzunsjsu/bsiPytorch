#pragma once

#include <vector>
#include <cstdint>

#include "../../../bsiCPP/bsi/BsiVector.hpp"
#include "../../../bsiCPP/bsi/hybridBitmap/hybridbitmap.h"

template <typename U>
inline void hb_to_verbatim_words_gpu_helper(const HybridBitmap<U>& hb,
                                            int64_t rows,
                                            std::vector<U>& out_words) {
    const int word_bits = 8 * sizeof(U);
    const int64_t W = (rows + word_bits - 1) / word_bits;
    out_words.clear();
    out_words.resize(static_cast<size_t>(W), static_cast<U>(0));

    // Fast path for verbatim storage: words are already literal.
    if (hb.verbatim) {
        const size_t n = static_cast<size_t>(W);
        for (size_t i = 0; i < n; ++i) {
            out_words[i] = hb.getWord(i);
        }
        return;
    }

    // Decompress EWAH-encoded buffer into W literal words using the raw iterator.
    auto it = hb.raw_iterator();
    size_t out_idx = 0;
    while (it.hasNext() && out_idx < static_cast<size_t>(W)) {
        auto& rle = it.next();
        const U run_len = rle.getRunningLength();
        const bool run_bit = rle.getRunningBit();
        // Emit run words
        const U run_word = run_bit ? static_cast<U>(~static_cast<U>(0)) : static_cast<U>(0);
        for (U k = 0; k < run_len && out_idx < static_cast<size_t>(W); ++k) {
            out_words[out_idx++] = run_word;
        }
        // Emit literal words following the RLW
        const U lit_words = rle.getNumberOfLiteralWords();
        const U* dw = it.dirtyWords();
        for (U k = 0; k < lit_words && out_idx < static_cast<size_t>(W); ++k) {
            out_words[out_idx++] = dw[k];
        }
    }
    // Pad with zeros if needed
    while (out_idx < static_cast<size_t>(W)) {
        out_words[out_idx++] = static_cast<U>(0);
    }
}

template <typename U>
inline void bsi_flatten_words_gpu_helper(const BsiVector<U>& vec,
                                         std::vector<U>& out,
                                         int& S,
                                         int& W) {
    const int word_bits = 8 * static_cast<int>(sizeof(U));
    const int64_t rows = vec.getNumberOfRows();
    W = static_cast<int>((rows + word_bits - 1) / word_bits);
    S = vec.getNumberOfSlices();
    out.clear();
    out.resize(static_cast<size_t>(S) * static_cast<size_t>(W), static_cast<U>(0));

    std::vector<U> tmp;
    for (int s = 0; s < S; ++s) {
        hb_to_verbatim_words_gpu_helper(vec.bsi[s], rows, tmp);
        std::copy(tmp.begin(),
                  tmp.end(),
                  out.begin() + static_cast<size_t>(s) * static_cast<size_t>(W));
    }
}
