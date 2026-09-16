#include "s2_sampler.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

static std::vector<std::pair<float, int32_t>> reference_order(const std::vector<float> & logits) {
    std::vector<std::pair<float, int32_t>> items;
    for (size_t i = 0; i < logits.size(); ++i) {
        if (logits[i] != -std::numeric_limits<float>::infinity())
            items.push_back({logits[i], static_cast<int32_t>(i)});
    }
    std::sort(items.begin(), items.end(), [](const auto & a, const auto & b) {
        return a.first > b.first;
    });
    return items;
}

static void expect_token(const std::vector<float> & logits, const s2::SamplerParams & params,
                         int32_t expected, const char * message) {
    if (s2::sample_token(logits.data(), static_cast<int32_t>(logits.size()), params) != expected)
        throw std::runtime_error(message);
}

int main() {
    s2::SamplerParams greedy;
    greedy.temperature = 0.0f;
    greedy.top_k = 1;
    std::vector<float> logits(4096);
    for (size_t i = 0; i < logits.size(); ++i) logits[i] = -1.0f - float(i) / 4096;
    logits[2027] = 3.25f;
    logits[18] = 0.0f;
    logits[31] = -0.0f;
    expect_token(logits, greedy, 2027, "positive/negative ordering changed");
    logits[2027] = -0.25f;
    logits[18] = logits[31] = -2.0f;
    expect_token(logits, greedy, 2027, "negative ordering changed");

    for (size_t i = 0; i < logits.size(); ++i)
        logits[i] = -float(i + 1) * std::numeric_limits<float>::denorm_min();
    logits[251] = std::numeric_limits<float>::denorm_min();
    logits[252] = 0.0f;
    logits[253] = -0.0f;
    expect_token(logits, greedy, 251, "subnormal ordering changed");

    // Equal-valued tokens must retain the previous comparison sort's selected
    // identity, not acquire stable radix order (including signed-zero ties).
    for (size_t i = 0; i < logits.size(); ++i) logits[i] = i % 2 ? -0.0f : 0.0f;
    expect_token(logits, greedy, reference_order(logits).front().second,
                 "tied greedy identity changed");

    // A tie crossing top-k must not substitute a different token. Masked
    // entries also make restoring sorted indices instead of original IDs fail.
    for (size_t i = 0; i < logits.size(); ++i)
        logits[i] = i % 3 ? -100.0f : -std::numeric_limits<float>::infinity();
    logits[77] = 1.0f;
    logits[99] = logits[2501] = 0.9f;
    const auto order = reference_order(logits);
    s2::SamplerParams tied;
    tied.top_k = 2;
    tied.top_p = 1.0f;
    for (int i = 0; i < 512; ++i) {
        const auto token = s2::sample_token(logits.data(), static_cast<int32_t>(logits.size()), tied);
        if (token != order[0].second && token != order[1].second)
            throw std::runtime_error("top-k boundary tie changed eligible token IDs");
    }

    std::fill(logits.begin(), logits.end(), -100.0f);
    logits[1234] = 2.0f;
    logits[3333] = 1.0f;
    logits[2038] = 0.0f;
    s2::SamplerParams nucleus;
    nucleus.top_k = 3;
    nucleus.top_p = 0.7f;
    nucleus.temperature = 10.0f;
    for (int i = 0; i < 32; ++i)
        expect_token(logits, nucleus, 1234, "top-p ordering or crossing-token exclusion changed");

    // A cutoff one float below the second cumulative probability exposes a
    // changed full-vocabulary reduction order, even when top-k is only two.
    for (size_t i = 0; i < logits.size(); ++i) logits[i] = float(i) * 0.00001f;
    const auto sorted = reference_order(logits);
    float sum = 0.0f;
    for (const auto & item : sorted) sum += std::exp(item.first - sorted.front().first);
    float cumulative = std::exp(sorted[0].first - sorted[0].first) / sum;
    cumulative += std::exp(sorted[1].first - sorted[0].first) / sum;
    nucleus.top_k = 2;
    nucleus.top_p = std::nextafter(cumulative, 0.0f);
    for (int i = 0; i < 32; ++i)
        expect_token(logits, nucleus, sorted.front().second, "nucleus boundary rounding changed");

    std::fill(logits.begin(), logits.end(), -std::numeric_limits<float>::infinity());
    expect_token(logits, nucleus, 0, "all-masked vocabulary changed");
    std::cout << "sampler ordering, ties, masks and nucleus boundaries passed\n";
}
