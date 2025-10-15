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
    // Read literal words directly via getWord, which returns the decompressed
    // literal word at index, regardless of hybrid/compressed storage.
    for (int64_t i = 0; i < W; ++i) {
        out_words[static_cast<size_t>(i)] = hb.getWord(static_cast<size_t>(i));
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
