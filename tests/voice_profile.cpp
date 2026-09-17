#include "../include/s2_voice.h"
#include "../third_party/filesystem.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fs = ghc::filesystem;

static void require(bool value) {
    if (!value) throw std::runtime_error("voice profile persistence contract failed");
}

template<class T>
static void field(std::vector<char> & bytes, size_t offset, T value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

int main() {
    char directory[] = "/tmp/s2-voice-profile-XXXXXX";
    if (!mkdtemp(directory)) return 2;
    struct Cleanup {
        const char * path;
        ~Cleanup() { fs::remove_all(path); }
    } cleanup{directory};
    try {
        const auto path = (fs::path(directory) / "reference.s2voice").string();
        s2::VoiceProfile profile;
        profile.transcript = "synthetic reference";
        profile.num_codebooks = 2;
        profile.T_prompt = 3;
        profile.codebook_size = 32;
        profile.codes = {0, 1, 31, 3, 4, 5};
        require(profile.save(path));
        const auto restored = s2::VoiceProfile::load(path);
        require(restored.codes == profile.codes && restored.transcript == profile.transcript &&
                restored.T_prompt == profile.T_prompt && restored.is_compatible(2, 32, 44100));
        require(!restored.is_compatible(2, 32, 24000));
        std::ifstream original(path, std::ios::binary);
        const std::vector<char> good((std::istreambuf_iterator<char>(original)), {});
        original.close();
        auto rejected = [&](const std::vector<char> & bytes) {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            out.close();
            try { (void) s2::VoiceProfile::load(path); }
            catch (const std::runtime_error &) { return; }
            throw std::runtime_error("corrupt voice profile was accepted");
        };
        rejected(std::vector<char>(good.begin(), good.begin() + 27));
        rejected(std::vector<char>(good.begin(), good.end() - 1));
        auto bad = good;
        field(bad, 28, std::numeric_limits<uint64_t>::max());
        rejected(bad); // Length must be bounded before allocation.
        bad = good;
        field(bad, 16, int32_t(4));
        rejected(bad); // Payload shape differs from book/frame dimensions.
        bad.assign(good.begin(), good.end() - 1);
        field(bad, 36, uint64_t(profile.codes.size()*sizeof(int32_t) - 1));
        rejected(bad); // Partial int32 payload.
        bad = good;
        field(bad, bad.size() - sizeof(int32_t), int32_t(32));
        rejected(bad); // A cache must not introduce out-of-vocabulary codes.
        bad = good;
        bad[44] = '\0';
        rejected(bad); // Embedded NUL changes the C ABI transcript.
        bad = good;
        bad.push_back('x');
        rejected(bad); // No unaccounted trailing bytes.
        profile.codes.pop_back();
        require(!profile.save(path));
        std::cout << "Voice profile persistence and corruption boundaries passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
