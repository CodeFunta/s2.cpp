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
    optimizer_alias_ordering(backend);
    ggml_backend_free(backend);
    std::cout<<"Metal padding, convolution and Q8 SwiGLU parity passed\n";
}
