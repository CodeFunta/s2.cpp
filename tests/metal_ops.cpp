#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-metal.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
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
    ggml_backend_free(backend);
    std::cout<<"Metal padding and convolution parity passed\n";
}
