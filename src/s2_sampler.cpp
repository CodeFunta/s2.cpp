#include "../include/s2_sampler.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <cstring>

namespace s2 {

static void apply_softmax(std::vector<float> & probs, float temp = 1.0f) {
    if (probs.empty()) return;
    float max_val = *std::max_element(probs.begin(), probs.end());
    float sum = 0.0f;
    for (float & p : probs) {
        if (temp > 0.0f) {
            p = std::exp((p - max_val) / temp);
        } else {
            p = (p == max_val) ? 1.0f : 0.0f;
        }
        sum += p;
    }
    if (sum > 0) {
        for (float & p : probs) p /= sum;
    }
}

static std::vector<float> softmax_from_sorted_logits(const std::vector<std::pair<float, int32_t>> & items) {
    std::vector<float> probs(items.size(), 0.0f);
    if (items.empty()) return probs;

    const float max_val = items.front().first;
    float sum = 0.0f;
    for (size_t i = 0; i < items.size(); ++i) {
        probs[i] = std::exp(items[i].first - max_val);
        sum += probs[i];
    }

    if (sum > 0.0f) {
        for (float & p : probs) p /= sum;
    }
    return probs;
}

static void sort_logits(std::vector<std::pair<float, int32_t>> & items,
                        const float * logits, int32_t vocab_size, int32_t top_k) {
    const auto descending = [](const auto & a, const auto & b) {
        return a.first > b.first;
    };
    // Histogram setup costs more than comparison sorting for small vocabularies.
    if (items.size() < 2048 ||
        !std::all_of(items.begin(), items.end(), [](const auto & v) {
            return std::isfinite(v.first);
        })) {
        std::sort(items.begin(), items.end(), descending);
        return;
    }

    static_assert(sizeof(float) == sizeof(uint32_t) &&
                  std::numeric_limits<float>::is_iec559, "radix keys require IEEE binary32");
    const auto key = [](float value) {
        uint32_t bits;
        if (value == 0.0f) value = 0.0f; // Both signed zeros compare equal.
        std::memcpy(&bits, &value, sizeof(bits));
        return ~(bits ^ ((bits >> 31) ? UINT32_MAX : 0x80000000u));
    };
    // Sort every value: the existing softmax must accumulate in the same order,
    // including the tail that top-k will discard.
    std::vector<std::pair<float, int32_t>> scratch(items.size());
    for (unsigned shift : {0u, 11u, 22u}) {
        uint32_t counts[2048] = {};
        const unsigned mask = shift == 22 ? 1023u : 2047u;
        for (const auto & item : items) ++counts[(key(item.first) >> shift) & mask];
        uint32_t offset = 0;
        for (unsigned bin = 0; bin <= mask; ++bin) {
            const uint32_t count = counts[bin];
            counts[bin] = offset;
            offset += count;
        }
        for (const auto & item : items)
            scratch[counts[(key(item.first) >> shift) & mask]++] = item;
        items.swap(scratch);
    }

    const size_t k = top_k > 0 ? std::min(size_t(top_k), items.size()) : items.size();
    for (size_t i = 1; i < items.size() && i <= k; ++i) {
        if (items[i - 1].first != items[i].first) continue;
        // A tie within or across top-k can change the selected token IDs.
        // Restore the ORIGINAL input before using the existing unstable sort.
        items.clear();
        for (int32_t j = 0; j < vocab_size; ++j) {
            if (logits[j] != -std::numeric_limits<float>::infinity())
                items.push_back({logits[j], j});
        }
        std::sort(items.begin(), items.end(), descending);
        return;
    }
}

int32_t sample_token(const float * logits, int32_t vocab_size, const SamplerParams & params) {
    if (vocab_size <= 0) return 0;

    std::vector<std::pair<float, int32_t>> items;
    items.reserve(vocab_size);
    for (int32_t i = 0; i < vocab_size; ++i) {
        // Masked vocabulary entries have zero probability. Do not sort or
        // exponentiate them for each semantic token.
        if (logits[i] != -std::numeric_limits<float>::infinity())
            items.push_back({logits[i], i});
    }
    if (items.empty()) return 0;

    sort_logits(items, logits, vocab_size, params.top_k);

    const int32_t candidates = static_cast<int32_t>(items.size());
    const int32_t k = params.top_k > 0 ? std::min(params.top_k, candidates) : candidates;
    const float top_p = std::clamp(params.top_p, 0.0f, 1.0f);
    const std::vector<float> sorted_probs = softmax_from_sorted_logits(items);

    std::vector<std::pair<float, int32_t>> filtered;
    filtered.reserve(k);

    float cumsum = 0.0f;
    for (int32_t i = 0; i < (int32_t)items.size(); ++i) {
        cumsum += sorted_probs[i];
        const bool remove_for_top_k = (i >= k);
        const bool remove_for_top_p = (i > 0 && cumsum > top_p);
        if (remove_for_top_k || remove_for_top_p) {
            continue;
        }
        filtered.push_back(items[i]);
    }

    if (filtered.empty()) {
        filtered.push_back(items.front());
    }

    if (params.temperature <= 0.0f) {
        return filtered[0].second;
    }

    std::vector<float> probs(filtered.size());
    for (size_t i = 0; i < filtered.size(); ++i) {
        probs[i] = filtered[i].first;
    }
    apply_softmax(probs, params.temperature);

    float sum_p = 0.0f;
    for (float p : probs) sum_p += p;
    if (sum_p <= 0.0f) {
        return filtered[0].second;
    }
    for (float & p : probs) p /= sum_p;

    thread_local static std::mt19937 gen(std::random_device{}());
    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());

    const int32_t sampled_idx = dist(gen);
    return filtered[sampled_idx].second;
}

}
