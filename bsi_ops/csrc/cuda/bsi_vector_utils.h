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
    out_words.reserve(W);

    HybridBitmapRawIterator<U> it = hb.raw_iterator();
    HybridBitmap<U> tmp(true);
    size_t written = 0;
    while (it.hasNext() && written < static_cast<size_t>(W)) {
        auto& brlw = it.next();
        size_t before = tmp.buffer.size();
        size_t just = brlw.dischargeDecompressed(tmp, static_cast<size_t>(W) - written);
        written += just;
        if (tmp.buffer.size() == before && just == 0) {
            break;
        }
    }
    out_words.assign(tmp.buffer.begin(), tmp.buffer.end());
    if (out_words.size() < static_cast<size_t>(W)) {
        out_words.resize(static_cast<size_t>(W), static_cast<U>(0));
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
