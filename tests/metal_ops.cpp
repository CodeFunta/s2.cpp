#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-metal.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

static void transpose_conv(ggml_backend_t backend, int channels, int length, int kernel, int stride, ggml_type type) {
    auto *ctx = ggml_init({1 << 20, nullptr, true});
    const int outputs = channels == 1536 ? 768 : 5;
    auto *weights = ggml_new_tensor_3d(ctx, type, kernel, outputs, channels);
    auto *input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, length, channels);
    auto *result = ggml_conv_transpose_1d(ctx, weights, input, stride, 0, 1);
    auto *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) throw std::runtime_error("allocation failed");
    std::vector<float> w(kernel*outputs*channels), x(length*channels), y(ggml_nelements(result));
    for (size_t i=0;i<w.size();++i) w[i]=0.1f*std::sin(float(i)*0.13f);
    for (size_t i=0;i<x.size();++i) x[i]=0.2f*std::cos(float(i)*0.17f);
    if(type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(w.size());
        for(size_t i=0;i<w.size();++i) {half[i]=ggml_fp32_to_fp16(w[i]); w[i]=ggml_fp16_to_fp32(half[i]);}
        ggml_backend_tensor_set(weights,half.data(),0,half.size()*2);
    } else ggml_backend_tensor_set(weights,w.data(),0,w.size()*4);
    ggml_backend_tensor_set(input,x.data(),0,x.size()*4);
    auto begin=std::chrono::steady_clock::now();
    if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
    auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    ggml_backend_tensor_get(result,y.data(),0,y.size()*4);
    const int out_length=(length-1)*stride+kernel;
    double max_error=0;
    for(size_t p=0;p<y.size();p += channels==1536 ? 97 : 1) {
        int j=p%out_length, oc=p/out_length;
        double expected=0;
        for(int c=0;c<channels;c++) for(int i=0;i<length;i++) {
            int k=j-i*stride;
            if(k>=0 && k<kernel) expected+=double(w[(c*outputs+oc)*kernel+k])*x[c*length+i];
        }
        max_error=std::max(max_error,std::abs(double(y[p])-expected));
        if(!std::isfinite(y[p]) || std::abs(y[p]-expected)>2e-4*(1+std::abs(expected)))
            throw std::runtime_error("transpose convolution differs from scalar reference");
    }
    std::cout<<"conv channels="<<channels<<" length="<<length<<" kernel="<<kernel<<" stride="<<stride<<" type="<<ggml_type_name(type)<<" max_error="<<max_error<<" ms="<<ms<<"\n";
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
}

static void left_pad(ggml_backend_t backend) {
    auto *ctx=ggml_init({1<<20,nullptr,true});
    auto *x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,5,3);
    auto *y=ggml_pad_ext(ctx,x,3,2,0,0,0,0,0,0);
    auto *g=ggml_new_graph(ctx); ggml_build_forward_expand(g,y);
    auto b=ggml_backend_alloc_ctx_tensors(ctx,backend);
    std::vector<float> input(15),output(30);
    for(int i=0;i<15;i++) input[i]=float(i+1);
    ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
    if(ggml_backend_graph_compute(backend,g)!=GGML_STATUS_SUCCESS) throw std::runtime_error("pad compute failed");
    ggml_backend_tensor_get(y,output.data(),0,output.size()*4);
    for(int row=0;row<3;row++) for(int col=0;col<10;col++) {
        float expected=col>=3 && col<8 ? input[row*5+col-3] : 0.f;
        if(output[row*10+col]!=expected) throw std::runtime_error("left pad changed samples");
    }
    ggml_backend_buffer_free(b); ggml_free(ctx);
}

static void swiglu_q8_regression(ggml_backend_t backend) {
    constexpr int64_t k = 1024;
    constexpr int64_t rows = 7;
    for (int64_t ncols : { int64_t(1), int64_t(2) }) {
        for (int scenario = 0; scenario < (ncols == 1 ? 4 : 1); ++scenario) {
            const bool requested_gate = scenario == 1;
            const bool add_consumer = scenario == 2;
            const bool distinct_up_input = scenario == 3;
            auto *ctx = ggml_init({1 << 22, nullptr, true});
            if (!ctx) throw std::runtime_error("SwiGLU context allocation failed");

            auto *wg = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, rows);
            auto *wu = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, rows);
            auto *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, ncols);
            auto *x_up = distinct_up_input ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, ncols) : x;
            auto *ref_x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, ncols);
            auto *ref_x_up = distinct_up_input ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, ncols) : ref_x;

            std::vector<float> gate_weights(k * rows), up_weights(k * rows);
            std::vector<float> input(k * ncols), up_input(k * ncols);
            for (size_t i = 0; i < gate_weights.size(); ++i) {
                gate_weights[i] = 0.11f * std::sin(float(i) * 0.017f) + 0.03f;
                up_weights[i] = -0.09f * std::cos(float(i) * 0.023f) + 0.02f;
            }
            for (size_t i = 0; i < input.size(); ++i) {
                input[i] = 0.17f + 0.13f * std::sin(float(i) * 0.031f);
                up_input[i] = -0.21f + 0.08f * std::cos(float(i) * 0.019f);
            }
            std::vector<uint8_t> qgate(ggml_nbytes(wg)), qup(ggml_nbytes(wu));
            ggml_quantize_chunk(GGML_TYPE_Q8_0, gate_weights.data(), qgate.data(), 0, rows, k, nullptr);
            ggml_quantize_chunk(GGML_TYPE_Q8_0, up_weights.data(), qup.data(), 0, rows, k, nullptr);

            auto *ref_gate = ggml_mul_mat(ctx, wg, ref_x);
            auto *ref_up = ggml_mul_mat(ctx, wu, ref_x_up);
            auto *ref_result = ggml_swiglu_split(ctx, ref_gate, ref_up);
            auto *ref_add = add_consumer ? ggml_add(ctx, ref_gate, ref_up) : nullptr;
            ggml_set_output(ref_gate);
            ggml_set_output(ref_up);
            ggml_set_output(ref_result);
            if (ref_add) ggml_set_output(ref_add);

            auto *candidate_gate = ggml_mul_mat(ctx, wg, x);
            auto *candidate_up = ggml_mul_mat(ctx, wu, x_up);
            auto *candidate_result = ggml_swiglu_split(ctx, candidate_gate, candidate_up);
            ggml_set_output(candidate_result);
            if (requested_gate) {
                ggml_set_output(candidate_gate);
            }
            auto *candidate_add = add_consumer ? ggml_add(ctx, candidate_gate, candidate_up) : nullptr;
            if (candidate_add) ggml_set_output(candidate_add);
            auto *graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, ref_result);
            if (ref_add) ggml_build_forward_expand(graph, ref_add);
            ggml_build_forward_expand(graph, candidate_result);
            if (candidate_add) ggml_build_forward_expand(graph, candidate_add);
            auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buffer) throw std::runtime_error("SwiGLU allocation failed");
            ggml_backend_tensor_set(wg, qgate.data(), 0, qgate.size());
            ggml_backend_tensor_set(wu, qup.data(), 0, qup.size());
            std::vector<float> unwritten(rows * ncols, std::numeric_limits<float>::quiet_NaN());
            ggml_backend_tensor_set(candidate_gate, unwritten.data(), 0, unwritten.size() * sizeof(float));
            ggml_backend_tensor_set(candidate_up, unwritten.data(), 0, unwritten.size() * sizeof(float));
            ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
            ggml_backend_tensor_set(ref_x, input.data(), 0, input.size() * sizeof(float));
            if (distinct_up_input) {
                ggml_backend_tensor_set(x_up, up_input.data(), 0, up_input.size() * sizeof(float));
                ggml_backend_tensor_set(ref_x_up, up_input.data(), 0, up_input.size() * sizeof(float));
            }
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("SwiGLU compute failed");

            const size_t result_size = size_t(ggml_nelements(ref_result));
            std::vector<float> expected(result_size), actual(result_size);
            ggml_backend_tensor_get(ref_result, expected.data(), 0, expected.size() * sizeof(float));
            ggml_backend_tensor_get(candidate_result, actual.data(), 0, actual.size() * sizeof(float));
            for (size_t i = 0; i < result_size; ++i) {
                if (!std::isfinite(expected[i]) || !std::isfinite(actual[i]) || expected[i] != actual[i])
                    throw std::runtime_error("Q8 SwiGLU result differs from unfused reference");
            }
            if (requested_gate) {
                std::vector<float> expected_gate(ggml_nelements(ref_gate)), actual_gate(ggml_nelements(candidate_gate));
                ggml_backend_tensor_get(ref_gate, expected_gate.data(), 0, expected_gate.size() * sizeof(float));
                ggml_backend_tensor_get(candidate_gate, actual_gate.data(), 0, actual_gate.size() * sizeof(float));
                for (size_t i = 0; i < expected_gate.size(); ++i) {
                    if (!std::isfinite(expected_gate[i]) || !std::isfinite(actual_gate[i]) || expected_gate[i] != actual_gate[i])
                        throw std::runtime_error("requested Q8 SwiGLU gate output differs");
                }
            }
            if (candidate_add) {
                std::vector<float> expected_add(ggml_nelements(ref_add)), actual_add(ggml_nelements(candidate_add));
                ggml_backend_tensor_get(ref_add, expected_add.data(), 0, expected_add.size() * sizeof(float));
                ggml_backend_tensor_get(candidate_add, actual_add.data(), 0, actual_add.size() * sizeof(float));
                for (size_t i = 0; i < expected_add.size(); ++i) {
                    if (!std::isfinite(expected_add[i]) || !std::isfinite(actual_add[i]) || actual_add[i] != expected_add[i])
                        throw std::runtime_error("unmarked Q8 SwiGLU projection consumer differs");
                }
            }
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
    }
}

static void matvec_add_q8_regression(ggml_backend_t backend) {
    constexpr int64_t k = 1024;
    constexpr int64_t rows = 257;
    for (int64_t ncols : {int64_t(1), int64_t(2)}) {
        for (int scenario = 0; scenario < (ncols == 1 ? 5 : 1); ++scenario) {
            const bool alias_residual = scenario == 1;
            const bool alias_projection = scenario == 2;
            const bool requested_projection = scenario == 3;
            const bool additional_consumer = scenario == 4;
            auto * ctx = ggml_init({1 << 24, nullptr, true});
            if (!ctx) throw std::runtime_error("Q8 residual context allocation failed");

            auto * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, rows);
            auto * ref_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, ncols);
            auto * candidate_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, ncols);
            auto * ref_residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, ncols);
            auto * candidate_residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, ncols);

            auto * ref_projection = ggml_mul_mat(ctx, weights, ref_input);
            auto * ref_sum = alias_residual
                ? ggml_add(ctx, ref_residual, ref_projection)
                : ggml_add(ctx, ref_projection, ref_residual);
            auto * ref_consumer = additional_consumer ? ggml_scale(ctx, ref_projection, 0.5f) : nullptr;
            ggml_set_output(ref_projection);
            ggml_set_output(ref_sum);
            if (ref_consumer) ggml_set_output(ref_consumer);

            auto * candidate_projection = ggml_mul_mat(ctx, weights, candidate_input);
            ggml_tensor * candidate_sum;
            if (alias_residual) {
                candidate_sum = ggml_add_inplace(ctx, candidate_residual, candidate_projection);
            } else if (alias_projection) {
                candidate_sum = ggml_add_inplace(ctx, candidate_projection, candidate_residual);
            } else {
                candidate_sum = ggml_add(ctx, candidate_projection, candidate_residual);
            }
            auto * candidate_consumer = additional_consumer ? ggml_scale(ctx, candidate_projection, 0.5f) : nullptr;
            if (requested_projection) ggml_set_output(candidate_projection);
            ggml_set_output(candidate_sum);
            if (candidate_consumer) ggml_set_output(candidate_consumer);

            auto * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, ref_sum);
            if (ref_consumer) ggml_build_forward_expand(graph, ref_consumer);
            ggml_build_forward_expand(graph, candidate_sum);
            if (candidate_consumer) ggml_build_forward_expand(graph, candidate_consumer);
            auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buffer) throw std::runtime_error("Q8 residual allocation failed");

            std::vector<float> weight_values(k * rows), input(k * ncols), residual(rows * ncols);
            for (size_t i = 0; i < weight_values.size(); ++i)
                weight_values[i] = 0.08f * std::sin(float(i) * 0.017f) - 0.02f;
            for (size_t i = 0; i < input.size(); ++i)
                input[i] = 0.13f * std::cos(float(i) * 0.031f) + 0.04f;
            for (size_t i = 0; i < residual.size(); ++i)
                residual[i] = 0.19f * std::sin(float(i) * 0.043f) - 0.07f;
            std::vector<uint8_t> quantized(ggml_nbytes(weights));
            ggml_quantize_chunk(GGML_TYPE_Q8_0, weight_values.data(), quantized.data(), 0, rows, k, nullptr);
            ggml_backend_tensor_set(weights, quantized.data(), 0, quantized.size());
            ggml_backend_tensor_set(ref_input, input.data(), 0, input.size() * sizeof(float));
            ggml_backend_tensor_set(candidate_input, input.data(), 0, input.size() * sizeof(float));
            ggml_backend_tensor_set(ref_residual, residual.data(), 0, residual.size() * sizeof(float));
            ggml_backend_tensor_set(candidate_residual, residual.data(), 0, residual.size() * sizeof(float));

            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("Q8 residual compute failed");

            std::vector<float> expected(residual.size()), actual(residual.size());
            ggml_backend_tensor_get(ref_sum, expected.data(), 0, expected.size() * sizeof(float));
            ggml_backend_tensor_get(candidate_sum, actual.data(), 0, actual.size() * sizeof(float));
            for (size_t i = 0; i < expected.size(); ++i) {
                if (!std::isfinite(expected[i]) || !std::isfinite(actual[i]) || actual[i] != expected[i])
                    throw std::runtime_error("Q8 fused residual result differs from unfused reference");
            }
            if (requested_projection) {
                std::vector<float> expected_projection(residual.size()), actual_projection(residual.size());
                ggml_backend_tensor_get(ref_projection, expected_projection.data(), 0, expected_projection.size() * sizeof(float));
                ggml_backend_tensor_get(candidate_projection, actual_projection.data(), 0, actual_projection.size() * sizeof(float));
                if (expected_projection != actual_projection)
                    throw std::runtime_error("requested Q8 projection output differs");
            }
            if (additional_consumer) {
                std::vector<float> expected_consumer(residual.size()), actual_consumer(residual.size());
                ggml_backend_tensor_get(ref_consumer, expected_consumer.data(), 0, expected_consumer.size() * sizeof(float));
                ggml_backend_tensor_get(candidate_consumer, actual_consumer.data(), 0, actual_consumer.size() * sizeof(float));
                if (expected_consumer != actual_consumer)
                    throw std::runtime_error("additional Q8 projection consumer differs");
            }

            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
    }
}

static void optimizer_alias_ordering(ggml_backend_t backend) {
    auto cpu = ggml_backend_cpu_init();
    if (!cpu) throw std::runtime_error("CPU fallback initialization failed");
    ggml_backend_t backends[] = {backend, cpu};
    for (bool fused : {false, true}) {
        auto * inputs = ggml_init({1 << 16, nullptr, true});
        auto * x = ggml_new_tensor_1d(inputs, GGML_TYPE_F32, 8);
        // Gaps prevent unrelated inputs from touching at interval endpoints.
        ggml_new_tensor_1d(inputs, GGML_TYPE_F32, 32);
        auto * z = ggml_new_tensor_1d(inputs, GGML_TYPE_F32, 4);
        ggml_new_tensor_1d(inputs, GGML_TYPE_F32, 32);
        auto * constant = ggml_new_tensor_1d(inputs, GGML_TYPE_F32, 4);
        ggml_new_tensor_1d(inputs, GGML_TYPE_F32, 32);
        auto * replacement = ggml_new_tensor_1d(inputs, GGML_TYPE_F32, 4);
        auto buffer = ggml_backend_alloc_ctx_tensors(inputs, backend);
        if (!buffer) throw std::runtime_error("alias input allocation failed");

        auto * ctx = ggml_init({1 << 20, nullptr, true});
        auto * alias = ggml_view_1d(ctx, x, 4, 4 * sizeof(float));
        // External input view: a VIEW operation would itself mask the hazard.
        alias->op = GGML_OP_NONE;
        auto * p = ggml_scale(ctx, z, 2.0f);
        auto * result = ggml_add(ctx, p, fused ? constant : alias);
        if (fused) result = ggml_mul(ctx, result, alias);
        auto * overwrite = ggml_cpy(ctx, replacement, alias);
        ggml_set_output(result);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);
        // No tensor edge orders this write after the earlier read of X.
        ggml_build_forward_expand(graph, overwrite);
        auto sched = ggml_backend_sched_new(backends, nullptr, 2, ggml_graph_size(graph), false, true);
        if (!ggml_backend_sched_alloc_graph(sched, graph)) throw std::runtime_error("alias graph allocation failed");
        for (int pass = 0; pass < 2; ++pass) {
            float original[8], z_values[4], constants[4], replacements[4], output[4], after[8];
            for (int i = 0; i < 8; ++i) original[i] = float(10 + 20 * pass + i);
            for (int i = 0; i < 4; ++i) {
                z_values[i] = float(i + 3);
                constants[i] = float(2 * i + 1);
                replacements[i] = float(100 + 20 * pass + i);
            }
            ggml_backend_tensor_set(x, original, 0, sizeof(original));
            ggml_backend_tensor_set(z, z_values, 0, sizeof(z_values));
            ggml_backend_tensor_set(constant, constants, 0, sizeof(constants));
            ggml_backend_tensor_set(replacement, replacements, 0, sizeof(replacements));
            if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("alias graph compute failed");
            ggml_backend_tensor_get(result, output, 0, sizeof(output));
            ggml_backend_tensor_get(x, after, 0, sizeof(after));
            for (int i = 0; i < 4; ++i) {
                const float expected = fused ? (2 * z_values[i] + constants[i]) * original[i + 4]
                                             : 2 * z_values[i] + original[i + 4];
                if (output[i] != expected || after[i] != original[i] || after[i + 4] != replacements[i])
                    throw std::runtime_error(fused ? "optimizer reordered a fused alias read" : "optimizer reordered an alias read");
            }
        }
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        ggml_backend_buffer_free(buffer);
        ggml_free(inputs);
    }
    ggml_backend_free(cpu);
}

static void flash_attention_tail_regression(ggml_backend_t backend) {
    struct Shape {
        int dim, keys, queries, batches;
        bool masked;
        float bias = 0.0f, softcap = 0.0f;
        bool sinks = false;
        bool unaligned = false;
    };
    const Shape cases[] = {
        {128, 2, 3, 2, false}, {128, 10, 2, 2, true},
        {128, 32, 1, 1, false}, {128, 33, 1, 2, true},
        {64, 65, 2, 2, false}, {96, 31, 3, 2, true},
        {128, 2049, 1, 1, true}, {128, 129, 20, 1, true},
        {128, 11, 3, 2, true, 0.7f, 2.0f, true},
        {128, 7, 2, 2, true, 0.0f, 0.0f, false, true},
    };
    constexpr int heads = 4, kv_heads = 2, mask_heads = 2;
    for (const auto & s : cases) {
        auto * ctx = ggml_init({1 << 22, nullptr, true});
        if (!ctx) throw std::runtime_error("attention context allocation failed");
        auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.dim, s.queries, heads, s.batches);
        // Interleave KV heads and leave poisoned rows outside the visible view.
        const int kv_dim = s.dim + (s.unaligned ? 1 : 0);
        auto make_kv = [&]() {
            if (!s.unaligned)
                return ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.dim, kv_heads, s.keys + 2, s.batches);
            auto * storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,
                int64_t(kv_dim)*kv_heads*(s.keys + 2)*s.batches + 1);
            return ggml_view_4d(ctx, storage, s.dim, kv_heads, s.keys + 2, s.batches,
                kv_dim*sizeof(float), kv_dim*kv_heads*sizeof(float),
                kv_dim*kv_heads*(s.keys + 2)*sizeof(float), sizeof(float));
        };
        auto * kb = make_kv();
        auto * vb = make_kv();
        auto view = [&](ggml_tensor * base) {
            return ggml_permute(ctx, ggml_view_4d(ctx, base, s.dim, kv_heads, s.keys, s.batches,
                base->nb[1], base->nb[2], base->nb[3], base->nb[2]), 0, 2, 1, 3);
        };
        auto * k = view(kb);
        auto * v = view(vb);
        auto * mask = s.masked ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16,
            s.keys, s.queries, mask_heads, s.batches) : nullptr;
        const float scale = 1.0f/std::sqrt(float(s.dim));
        auto * result = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, s.bias, s.softcap);
        auto * sinks = s.sinks ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads) : nullptr;
        if (sinks) ggml_flash_attn_ext_add_sinks(result, sinks);
        ggml_flash_attn_ext_set_prec(result, GGML_PREC_F32);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) throw std::runtime_error("attention allocation failed");
        std::vector<float> qv(ggml_nelements(q));
        std::vector<float> kval(ggml_nbytes(kb)/sizeof(float), std::numeric_limits<float>::quiet_NaN());
        std::vector<float> vval(kval.size(), std::numeric_limits<float>::quiet_NaN());
        auto qi = [&](int d, int query, int head, int batch) {
            return d + s.dim*(query + s.queries*(head + heads*batch));
        };
        auto ki = [&](int d, int key, int head, int batch) {
            return d + kv_dim*(head + kv_heads*(key + 1 + (s.keys + 2)*batch));
        };
        for (size_t i = 0; i < qv.size(); ++i) qv[i] = float(int(i%19) - 9)/32.0f;
        for (int b = 0; b < s.batches; ++b)
            for (int h = 0; h < kv_heads; ++h)
                for (int t = 0; t < s.keys; ++t)
                    for (int d = 0; d < s.dim; ++d) {
                        kval[ki(d,t,h,b)] = float((d + 3*t + 5*h + 7*b)%23 - 11)/64.0f;
                        vval[ki(d,t,h,b)] = float((5*d + 7*t + 3*h + b)%29 - 14)/16.0f;
                    }
        auto masked = [&](int t, int query, int head, int batch) {
            return s.masked && ((head%mask_heads == 1 && query == 0) ||
                (t + query + head%mask_heads + batch)%5 == 0);
        };
        ggml_backend_tensor_set(q, qv.data(), 0, qv.size()*sizeof(float));
        ggml_backend_tensor_set(kb, kval.data(), 0, kval.size()*sizeof(float));
        ggml_backend_tensor_set(vb, vval.data(), 0, vval.size()*sizeof(float));
        const float sink_values[heads] = {-0.5f, 0.0f, 0.5f, 1.0f};
        if (sinks) ggml_backend_tensor_set(sinks, sink_values, 0, sizeof(sink_values));
        auto mask_value = [&](int t, int query, int head, int batch) {
            if (masked(t, query, head, batch)) return -INFINITY;
            return s.bias != 0.0f ? -float(t%3)/8.0f : 0.0f;
        };
        if (mask) {
            std::vector<ggml_fp16_t> mv(ggml_nelements(mask));
            for (int b = 0; b < s.batches; ++b)
                for (int h = 0; h < mask_heads; ++h)
                    for (int query = 0; query < s.queries; ++query)
                        for (int t = 0; t < s.keys; ++t)
                            mv[t + s.keys*(query + s.queries*(h + mask_heads*b))] =
                                ggml_fp32_to_fp16(mask_value(t,query,h,b));
            ggml_backend_tensor_set(mask, mv.data(), 0, mv.size()*sizeof(ggml_fp16_t));
        }
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("attention compute failed");
        std::vector<float> actual(ggml_nelements(result));
        ggml_backend_tensor_get(result, actual.data(), 0, actual.size()*sizeof(float));
        for (int b = 0; b < s.batches; ++b)
            for (int h = 0; h < heads; ++h)
                for (int query = 0; query < s.queries; ++query) {
                    std::vector<double> weights(s.keys, 0.0);
                    double denom = s.sinks ? std::exp(double(sink_values[h])) : 0.0;
                    for (int t = 0; t < s.keys; ++t) {
                        if (masked(t,query,h,b)) continue;
                        double dot = 0.0;
                        for (int d = 0; d < s.dim; ++d)
                            dot += double(qv[qi(d,query,h,b)])*kval[ki(d,t,h/2,b)];
                        double score = s.softcap != 0.0f ? s.softcap*std::tanh(dot*scale/s.softcap) : dot*scale;
                        const double slope = s.bias != 0.0f ? std::pow(2.0, -double(s.bias)*(h + 1)/heads) : 1.0;
                        score += mask_value(t,query,h,b)*slope;
                        weights[t] = std::exp(score);
                        denom += weights[t];
                    }
                    for (int d = 0; d < s.dim; ++d) {
                        double expected = 0.0;
                        for (int t = 0; t < s.keys; ++t)
                            expected += weights[t]*vval[ki(d,t,h/2,b)];
                        if (denom != 0.0) expected /= denom;
                        const float got = actual[d + s.dim*(h + heads*(query + s.queries*b))];
                        if (!(std::abs(double(got) - expected) <= 2e-4))
                            throw std::runtime_error("attention tail/mask/batch mismatch");
                    }
                }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
}

static void rope_cache_copy_regression(ggml_backend_t backend) {
    constexpr int head = 128, heads = 2;
    auto cpu = ggml_backend_cpu_init();
    if (!cpu) throw std::runtime_error("RoPE cache CPU backend initialization failed");
    ggml_backend_t backends[] = {backend, cpu};
    for (int scenario = 0; scenario < 4; ++scenario) {
        const bool exposed = scenario == 1;
        const bool shared = scenario == 2;
        const bool overlap = scenario == 3;
        // Multiple scheduling waves expose cross-row overlap hazards.
        const int tokens = overlap ? 2048 : 3, count = head * heads * tokens;
        auto * inputs = ggml_init({1 << 16, nullptr, true});
        auto * reference_input = ggml_new_tensor_3d(inputs, GGML_TYPE_F32, head, heads, tokens);
        auto * source_storage = ggml_new_tensor_1d(inputs, GGML_TYPE_F32, count + 2);
        auto * cache_storage = ggml_new_tensor_1d(inputs, GGML_TYPE_F32, count + 2);
        auto * reference_storage = ggml_new_tensor_1d(inputs, GGML_TYPE_F32, count + 2);
        auto * positions = ggml_new_tensor_1d(inputs, GGML_TYPE_I32, tokens);
        ggml_set_input(reference_input);
        ggml_set_input(source_storage);
        ggml_set_input(positions);
        auto buffer = ggml_backend_alloc_ctx_tensors(inputs, backend);
        if (!buffer) throw std::runtime_error("RoPE cache input allocation failed");

        auto * ctx = ggml_init({1 << 20, nullptr, true});
        const auto view = [&](ggml_tensor * storage, size_t offset) {
            return ggml_view_3d(ctx, storage, head, heads, tokens,
                head * sizeof(float), head * heads * sizeof(float), offset);
        };
        auto * source = view(source_storage, 0);
        auto * reference_slot = view(reference_storage, sizeof(float));
        auto * destination_storage = overlap ? source_storage : cache_storage;
        auto * slot = view(destination_storage, sizeof(float));
        const auto rotate = [&](ggml_tensor * input) {
            return ggml_rope_ext(ctx, input, positions, nullptr, head, 0, 512,
                10000.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        };
        auto * reference = rotate(reference_input);
        auto * candidate = rotate(source);
        ggml_set_output(reference); // Force the independent two-dispatch oracle.
        if (exposed) ggml_set_output(candidate);
        auto * consumer = shared ? ggml_scale(ctx, candidate, 0.5f) : nullptr;
        if (consumer) ggml_set_output(consumer);
        auto * graph = ggml_new_graph(ctx);
        // Destination VIEW metadata must not separate producer and copy.
        ggml_build_forward_expand(graph, reference_slot);
        ggml_build_forward_expand(graph, slot);
        ggml_build_forward_expand(graph, ggml_cpy(ctx, reference, reference_slot));
        ggml_build_forward_expand(graph, ggml_cpy(ctx, candidate, slot));
        if (consumer) ggml_build_forward_expand(graph, consumer);
        auto sched = ggml_backend_sched_new(backends, nullptr, 2, ggml_graph_size(graph), false, true);
        if (!ggml_backend_sched_alloc_graph(sched, graph))
            throw std::runtime_error("RoPE cache graph allocation failed");

        std::vector<float> values(count + 2, -7.25f), guards(count + 2, -7.25f), expected(count), actual(count + 2);
        for (int i = 0; i < count; ++i) values[i] = 0.2f * std::sin(float(i) * 0.13f);
        std::vector<int32_t> position_values(tokens);
        for (int t = 0; t < tokens; ++t) position_values[t] = 17 + 14 * t;
        ggml_backend_tensor_set(reference_input, values.data(), 0, count * sizeof(float));
        ggml_backend_tensor_set(source_storage, values.data(), 0, values.size() * sizeof(float));
        ggml_backend_tensor_set(cache_storage, guards.data(), 0, guards.size() * sizeof(float));
        ggml_backend_tensor_set(reference_storage, guards.data(), 0, guards.size() * sizeof(float));
        ggml_backend_tensor_set(positions, position_values.data(), 0, position_values.size() * sizeof(int32_t));
        if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("RoPE cache graph compute failed");
        ggml_backend_tensor_get(reference, expected.data(), 0, count * sizeof(float));
        ggml_backend_tensor_get(destination_storage, actual.data(), 0, actual.size() * sizeof(float));
        if (actual.front() != (overlap ? values.front() : guards.front()) || actual.back() != guards.back() ||
            !std::equal(expected.begin(), expected.end(), actual.begin() + 1))
            throw std::runtime_error("RoPE cache copy changed values or adjacent storage");
        if (exposed || shared) {
            std::vector<float> observed(count);
            ggml_backend_tensor_get(exposed ? candidate : consumer, observed.data(), 0, count * sizeof(float));
            for (int i = 0; i < count; ++i) {
                if (observed[i] != (exposed ? expected[i] : expected[i] * 0.5f))
                    throw std::runtime_error("RoPE cache fusion discarded an observable intermediate");
            }
        }
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        ggml_backend_buffer_free(buffer);
        ggml_free(inputs);
    }
    ggml_backend_free(cpu);
}

int main() {
    auto backend=ggml_backend_metal_init();
    if(!backend) return 2;
    left_pad(backend);
    for(auto type:{GGML_TYPE_F32,GGML_TYPE_F16}) {
        for(int channels:{1,7,32,33,128}) {
            transpose_conv(backend,channels,3,3,1,type);
            transpose_conv(backend,channels,17,4,2,type);
        }
        transpose_conv(backend,1536,8,16,8,type);
    }
    swiglu_q8_regression(backend);
    matvec_add_q8_regression(backend);
    optimizer_alias_ordering(backend);
    flash_attention_tail_regression(backend);
    rope_cache_copy_regression(backend);
    ggml_backend_free(backend);
    std::cout<<"Metal padding, convolution and Q8 fusion parity passed\n";
}
