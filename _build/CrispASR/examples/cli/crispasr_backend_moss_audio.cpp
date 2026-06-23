// crispasr_backend_moss_audio.cpp — adapter for MOSS-Audio-4B-Instruct.
//
// Pipeline: mel → 32L Whisper encoder (DeepStack taps at L8/16/24) →
// audio_adapter GatedMLP → masked_scatter into prompt embeds →
// 3× deepstack merger → inject at LM L0/L1/L2 → 36L Qwen3 decode.
//
// First audio-understanding (not just ASR) backend in CrispASR —
// supports transcription, audio QA, scene description, etc via prompt.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "moss_audio.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// GPT-2 byte decoder (same as qwen3 backend — shared BPE byte mapping)
std::vector<int>& byte_decoder() {
    static std::vector<int> dec(0x200, -1);
    static bool initialized = false;
    if (initialized)
        return dec;
    std::vector<int> bs, cs;
    for (int b = 0x21; b <= 0x7e; b++) {
        bs.push_back(b);
        cs.push_back(b);
    }
    for (int b = 0xa1; b <= 0xac; b++) {
        bs.push_back(b);
        cs.push_back(b);
    }
    for (int b = 0xae; b <= 0xff; b++) {
        bs.push_back(b);
        cs.push_back(b);
    }
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool present = false;
        for (int x : bs)
            if (x == b) {
                present = true;
                break;
            }
        if (!present) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }
    for (size_t i = 0; i < bs.size(); i++) {
        if ((size_t)cs[i] < dec.size())
            dec[cs[i]] = bs[i];
    }
    initialized = true;
    return dec;
}

std::string decode_token(const std::string& s) {
    auto& dec = byte_decoder();
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = s[i];
        int cp = 0, len = 1;
        if (c < 0x80) {
            cp = c;
            len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F;
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F;
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07;
            len = 4;
        } else {
            i++;
            continue;
        }
        if (i + len > s.size())
            break;
        for (int k = 1; k < len; k++)
            cp = (cp << 6) | (s[i + k] & 0x3F);
        i += len;
        if (cp >= 0 && cp < (int)dec.size() && dec[cp] >= 0)
            out.push_back((char)dec[cp]);
    }
    return out;
}

class MossAudioBackend : public CrispasrBackend {
public:
    MossAudioBackend() = default;
    ~MossAudioBackend() override { MossAudioBackend::shutdown(); }

    const char* name() const override { return "moss-audio"; }

    uint32_t capabilities() const override { return CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_PUNCTUATION_NATIVE; }

    bool init(const whisper_params& p) override {
        auto cp = moss_audio_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        ctx_ = moss_audio_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[moss-audio]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        if (p.seed > 0)
            moss_audio_set_seed(ctx_, (uint32_t)p.seed);
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        if (!ctx_)
            return {};

        // Use the prompt from params if set, otherwise default to transcription
        const char* prompt = "Transcribe this audio.";
        if (!params.prompt.empty()) {
            prompt = params.prompt.c_str();
        }

        char* result = moss_audio_process(ctx_, samples, n_samples, prompt);
        if (!result)
            return {};

        std::string text(result);
        free(result);

        // Build segment
        crispasr_segment seg;
        seg.text = text;
        seg.t0 = t_offset_cs;
        int64_t dur_cs = (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.t1 = t_offset_cs + dur_cs;
        return {seg};
    }

    void shutdown() override {
        if (ctx_) {
            moss_audio_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    moss_audio_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_moss_audio_backend() {
    return std::unique_ptr<CrispasrBackend>(new MossAudioBackend());
}
