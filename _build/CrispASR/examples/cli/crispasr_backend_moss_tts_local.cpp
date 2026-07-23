// crispasr_backend_moss_tts_local.cpp — CLI adapter for
// MOSS-TTS-Local-Transformer-v1.5 (MossTTSLocal, 4B).
//
// Two-GGUF runtime: the Qwen3-4B backbone (from --model) plus a companion
// MOSS-Audio-Tokenizer-v2 codec (from --codec-model, a sibling
// "<stem>-codec.gguf", or the registry companion). 48 kHz output (downmixed to
// mono). Voice cloning is not yet available (the v2 codec encoder is a
// follow-up), so no CAP_VOICE_CLONING.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "moss_tts_local.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

static bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static std::string dir_of(const std::string& p) {
    auto sep = p.find_last_of("/\\");
    return (sep == std::string::npos) ? std::string(".") : p.substr(0, sep);
}

// Look for a sibling codec GGUF next to the backbone. The converter writes
// "<stem>-codec.gguf"; the auto-download path drops both into the same dir.
static std::string discover_codec(const std::string& model_path) {
    const std::string ext = ".gguf";
    if (model_path.size() > ext.size() && model_path.compare(model_path.size() - ext.size(), ext.size(), ext) == 0) {
        std::string derived = model_path.substr(0, model_path.size() - ext.size()) + "-codec.gguf";
        if (file_exists(derived))
            return derived;
    }
    const std::string dir = dir_of(model_path);
    static const char* candidates[] = {
        "moss-tts-local-codec.gguf",
        "moss-tts-local-v1.5-codec.gguf",
        "moss-audio-tokenizer-v2.gguf",
    };
    for (const char* name : candidates) {
        std::string p = dir + "/" + name;
        if (file_exists(p))
            return p;
    }
    return "";
}

class MossTtsLocalBackend : public CrispasrBackend {
public:
    MossTtsLocalBackend() = default;
    ~MossTtsLocalBackend() override { MossTtsLocalBackend::shutdown(); }

    const char* name() const override { return "moss-tts-local"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_FLASH_ATTN; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[moss-tts-local]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        moss_tts_local_context_params cp = moss_tts_local_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        cp.flash_attn = p.flash_attn;
        ctx_ = moss_tts_local_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[moss-tts-local]: failed to load backbone '%s'\n", p.model.c_str());
            return false;
        }

        // Resolve the companion codec GGUF.
        std::string codec_path = p.tts_codec_model;
        if (!codec_path.empty() && codec_path != "auto" && codec_path != "default") {
            codec_path = crispasr_resolve_model_cli(codec_path, p.backend, p.no_prints, p.cache_dir, p.auto_download,
                                                    p.tts_codec_quant);
        } else {
            codec_path.clear();
        }
        if (codec_path.empty())
            codec_path = discover_codec(p.model);
        if (codec_path.empty()) {
            CrispasrRegistryEntry entry;
            if (crispasr_registry_lookup(p.backend, entry, p.tts_codec_quant) && !entry.companion_filename.empty()) {
                codec_path = crispasr_resolve_model_cli(entry.companion_filename, p.backend, p.no_prints, p.cache_dir,
                                                        p.auto_download, p.tts_codec_quant);
            }
        }
        if (codec_path.empty()) {
            fprintf(stderr, "crispasr[moss-tts-local]: no codec model found. Pass --codec-model PATH or place "
                            "<stem>-codec.gguf next to the backbone.\n");
            return false;
        }
        if (!moss_tts_local_set_codec_path(ctx_, codec_path.c_str())) {
            fprintf(stderr, "crispasr[moss-tts-local]: failed to load codec '%s'\n", codec_path.c_str());
            return false;
        }
        if (!p.no_prints)
            fprintf(stderr, "crispasr[moss-tts-local]: codec loaded from '%s'\n", codec_path.c_str());
        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty())
            return {};
        moss_tts_local_synth_params sp = moss_tts_local_synth_default_params();
        sp.seed = params.seed;
        if (params.temperature > 0.0f) {
            sp.text_temperature = params.temperature;
            sp.audio_temperature = params.temperature;
        }
        // Explicit-only: whisper_params default 512 vs moss-tts-local default 4096
        // — `> 0` would shrink every synthesis to 512 frames. Gate on the flag (#292).
        if (params.max_new_tokens_explicit)
            sp.max_new_frames = params.max_new_tokens;
        if (params.tts_max_speech_tokens >= 0)
            sp.max_audio_frames = params.tts_max_speech_tokens;
        if (params.tts_top_p >= 0.0f)
            sp.audio_top_p = params.tts_top_p;
        if (params.tts_top_k >= 0)
            sp.audio_top_k = params.tts_top_k;
        if (params.tts_repetition_penalty >= 0.0f)
            sp.audio_repetition_penalty = params.tts_repetition_penalty;
        std::string lang;
        if (!params.language.empty() && params.language != "auto") {
            lang = crispasr_iso_to_english_lang(params.language);
            sp.language = lang.c_str();
        }
        if (!params.tts_instruct.empty())
            sp.instruction = params.tts_instruct.c_str();

        int n = 0;
        float* pcm = moss_tts_local_synthesize(ctx_, text.c_str(), &sp, &n);
        if (!pcm || n <= 0) {
            free(pcm);
            return {};
        }
        std::vector<float> out(pcm, pcm + n);
        free(pcm);
        return out;
    }

    int tts_sample_rate() const override { return ctx_ ? moss_tts_local_sampling_rate(ctx_) : 48000; }

    void shutdown() override {
        if (ctx_) {
            moss_tts_local_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    moss_tts_local_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_moss_tts_local_backend() {
    return std::unique_ptr<CrispasrBackend>(new MossTtsLocalBackend());
}
