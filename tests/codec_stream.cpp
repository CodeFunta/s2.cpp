#include "s2_codec.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    s2::AudioCodec codec;
    if (!codec.load(argv[1], 0, s2::BackendType::Metal)) return 2;
    const int frames = argc > 2 ? std::stoi(argv[2]) : 161;
    const int books = codec.num_codebooks();
    std::vector<int32_t> codes(books * frames);
    for (int cb = 0; cb < books; ++cb) for (int t = 0; t < frames; ++t)
        codes[cb * frames + t] = (t * 17 + cb * 31) % 997;
    std::vector<float> reference;
    if (!codec.decode(codes.data(), frames, 8, reference)) return 3;
    if (argc > 3) {
        std::ofstream file(argv[3], std::ios::binary);
        file.write(reinterpret_cast<const char *>(reference.data()), reference.size() * sizeof(float));
    }
    // Repeated chunks after history saturation exercise retained graphs;
    // alternating shapes and the final tail exercise invalidation.
    const std::vector<int> chunks{1, 3, 8, 4, 4, 4, 4, 16};
    for (int pass = 0; pass < 2; ++pass) {
        codec.clear_decode_cache();
        std::vector<float> streamed;
        int offset = 0, chunk_index = pass;
        double total_ms = 0, max_ms = 0;
        while (offset < frames) {
            int n = std::min(chunks[chunk_index++ % chunks.size()], frames - offset);
            std::vector<int32_t> block(books * n);
            for (int cb = 0; cb < books; ++cb)
                std::copy_n(codes.data() + cb * frames + offset, n, block.data() + cb * n);
            std::vector<float> pcm;
            auto start = std::chrono::steady_clock::now();
            if (!codec.decode_stream(block.data(), n, 8, pcm)) return 4;
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            total_ms += ms; max_ms = std::max(max_ms, ms);
            if (pcm.size() != size_t(n) * codec.samples_per_code_frame())
                throw std::runtime_error("streamed sample count differs");
            streamed.insert(streamed.end(), pcm.begin(), pcm.end());
            offset += n;
        }
        if (streamed.size() != reference.size()) throw std::runtime_error("total sample count differs");
        double squared_error = 0, energy = 0, max_error = 0;
        for (size_t i = 0; i < reference.size(); ++i) {
            if (!std::isfinite(streamed[i])) throw std::runtime_error("nonfinite streaming output");
            double error = double(streamed[i]) - reference[i];
            squared_error += error * error; energy += double(reference[i]) * reference[i];
            max_error = std::max(max_error, std::abs(error));
        }
        double relative_rms = std::sqrt(squared_error / std::max(energy, 1e-30));
        std::cout << "pass=" << pass << " samples=" << streamed.size() << " max_error=" << max_error
                  << " relative_rms=" << relative_rms << " total_ms=" << total_ms << " max_chunk_ms=" << max_ms << std::endl;
        // Metal GEMM selects different F16 tiling for full and chunked shapes.
        // Bound waveform drift; missing history produces much larger discontinuities.
        if (max_error > 0.005 || relative_rms > 0.003)
            throw std::runtime_error("streaming codec differs from full decode");
    }
}
