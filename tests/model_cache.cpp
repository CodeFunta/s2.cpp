#include "s2_model.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Request {
    std::vector<float> hidden;
    std::vector<int32_t> prefix;
    std::vector<float> logits;
};

static void slow_cache_fusion_regression(const char * path) {
    const char * previous = std::getenv("GGML_METAL_FUSION_DISABLE");
    const bool had_previous = previous != nullptr;
    const std::string saved = previous ? previous : "";
    std::vector<s2::StepResult> reference;
    for (bool fused : {false, true}) {
        if (fused) unsetenv("GGML_METAL_FUSION_DISABLE");
        else setenv("GGML_METAL_FUSION_DISABLE", "1", 1);
        s2::SlowARModel model;
        if (!model.load(path, 0, s2::BackendType::Metal, -1) || !model.init_kv_cache(32))
            throw std::runtime_error("slow cache model initialization failed");
        const auto & hp = model.hparams();
        const int rows = hp.num_codebooks + 1;
        std::vector<int32_t> prompt(16 * rows, 0), token(rows, 3);
        for (int t = 0; t < 16; ++t) prompt[t * rows] = 100 + t;
        s2::StepResult state;
        for (int step = 0; step < 3; ++step) {
            token[0] = hp.semantic_begin_id + step;
            const bool ok = step == 0
                ? model.prefill_semantic(prompt, 16, 4, hp.semantic_begin_id - 33, state)
                : model.step_semantic(token, 4, hp.semantic_begin_id - 33, state);
            if (!ok) throw std::runtime_error("slow cache transition failed");
            if (!fused) reference.push_back(state);
            else if (state.hidden != reference[step].hidden || state.logits != reference[step].logits)
                throw std::runtime_error("fused cache write changed current or historical attention");
        }
    }
    if (had_previous) setenv("GGML_METAL_FUSION_DISABLE", saved.c_str(), 1);
    else unsetenv("GGML_METAL_FUSION_DISABLE");
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    s2::SlowARModel model;
    if (!model.load(argv[1], 0, s2::BackendType::Metal, -1)) return 3;
    if (!model.init_kv_cache(32)) return 3;
    const int rows = model.hparams().num_codebooks + 1;
    std::vector<int32_t> prompt(16 * rows, 0);
    for (int t = 0; t < 16; ++t) prompt[t * rows] = 100 + t;
    s2::StepResult state;
    if (!model.prefill_fast(prompt, 16, 4, state)) return 3;
    std::vector<float> hidden = state.hidden;
    std::vector<Request> requests;
    std::vector<int32_t> prefix;
    for (int n = 1; n < model.hparams().num_codebooks; ++n) {
        prefix.push_back((n * 17) % model.hparams().codebook_size);
        requests.push_back({hidden, prefix, {}});
    }
    requests.push_back({hidden, {3, 5}, {}});
    requests.push_back({hidden, {4, 5, 7}, {}});
    hidden[0] += 1.0f;
    requests.push_back({hidden, {4, 5, 7, 11}, {}});
    requests.push_back({hidden, {}, {}});
    // Revisit every retained graph with different hidden state and token inputs.
    prefix.clear();
    for (int n = 1; n < model.hparams().num_codebooks; ++n) {
        prefix.push_back((n * 19) % model.hparams().codebook_size);
        requests.push_back({hidden, prefix, {}});
    }
    for (auto & r : requests)
        if (!model.fast_decode(r.hidden, r.prefix, 4, r.logits)) return 4;
    double max_error = 0, max_relative_rms = 0, max_total_variation = 0;
    for (const auto & r : requests) {
        model.clear_kv_cache();
        std::vector<float> full;
        if (!model.fast_decode(r.hidden, r.prefix, 4, full)) return 5;
        if (full.size() != r.logits.size()) throw std::runtime_error("logit size differs");
        double squared_error = 0, energy = 0;
        for (size_t i = 0; i < full.size(); ++i) {
            if (!std::isfinite(r.logits[i])) throw std::runtime_error("nonfinite cached logits");
            double error = double(r.logits[i]) - full[i];
            max_error = std::max(max_error, std::abs(error));
            squared_error += error * error; energy += double(full[i]) * full[i];
        }
        max_relative_rms = std::max(max_relative_rms, std::sqrt(squared_error / std::max(energy, 1e-30)));
        const auto expected_best = std::max_element(full.begin(), full.end());
        const auto actual_best = std::max_element(r.logits.begin(), r.logits.end());
        if (expected_best - full.begin() != actual_best - r.logits.begin())
            throw std::runtime_error("cached greedy choice differs");
        double expected_sum = 0, actual_sum = 0;
        for (size_t i = 0; i < full.size(); ++i) {
            expected_sum += std::exp((double(full[i]) - *expected_best) / 0.8);
            actual_sum += std::exp((double(r.logits[i]) - *actual_best) / 0.8);
        }
        double variation = 0;
        for (size_t i = 0; i < full.size(); ++i)
            variation += std::abs(std::exp((double(full[i]) - *expected_best) / 0.8) / expected_sum -
                                  std::exp((double(r.logits[i]) - *actual_best) / 0.8) / actual_sum);
        max_total_variation = std::max(max_total_variation, variation / 2);
    }
    std::cout << "requests=" << requests.size() << " max_error=" << max_error
              << " max_relative_rms=" << max_relative_rms
              << " max_probability_total_variation=" << max_total_variation << std::endl;
    // F32 precision keeps quantized Metal accumulation consistent across
    // full-prefix and one-query shapes; tolerate only float rounding noise.
    if (max_error > 1e-4 || max_total_variation > 1e-5)
        throw std::runtime_error("cached logits differ from full recomputation");
    const auto & hp = model.hparams();
    for (int32_t eos : {0, hp.semantic_begin_id - 33, hp.vocab_size - 1}) {
        auto compare = [&](const s2::StepResult & full, const s2::StepResult & compact) {
            const int32_t begin = std::min(hp.semantic_begin_id, eos);
            const int32_t end = std::max(hp.semantic_end_id, eos);
            if (full.logits_offset != 0 || full.logits.size() != size_t(hp.vocab_size) ||
                compact.logits_offset != begin || compact.logits.size() != size_t(end - begin + 1) ||
                full.hidden != compact.hidden)
                throw std::runtime_error("semantic projection result shape/hidden differs");
            for (size_t i = 0; i < compact.logits.size(); ++i)
                if (compact.logits[i] != full.logits[begin + i])
                    throw std::runtime_error("semantic/EOS logits differ from full projection");
        };
        std::vector<int32_t> token(rows, 3);
        token[0] = hp.semantic_begin_id + 3;
        s2::StepResult full_prefill, full_step, compact_prefill, compact_step;
        model.clear_kv_cache();
        if (!model.init_kv_cache(32)) return 6;
        if (!model.prefill_fast(prompt, 16, 4, full_prefill) ||
            !model.step(token, 4, full_step)) return 6;
        model.clear_kv_cache();
        if (!model.init_kv_cache(32)) return 7;
        if (!model.prefill_semantic(prompt, 16, 4, eos, compact_prefill) ||
            !model.step_semantic(token, 4, eos, compact_step)) return 7;
        compare(full_prefill, compact_prefill);
        compare(full_step, compact_step);
    }
    std::cout << "semantic/EOS full-projection parity: exact" << std::endl;
    slow_cache_fusion_regression(argv[1]);
}
