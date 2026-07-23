// moss_tts_local_codec.cpp — MOSS-Audio-Tokenizer-v2 codec (decode).
//
// From-scratch port of the decode path of modeling_moss_audio_tokenizer.py
// (read line-by-line, HARD RULE #1). Shares the transformer-block / weight-norm
// / patch-upsample machinery with the v1 codec (src/moss_tts_codec.cpp) but
// adds: 12 (not 32) quantizers, hop 3840 @48 kHz, 6 decoder stages with an
// input_proj AND output_proj on EVERY stage, GELU-erf FFN, and a stereo
// (channel-interleaved) output. See moss_tts_local_codec.h for the pipeline.
//
// NOT yet parity-checked end-to-end: the ASR round-trip on Kaggle (HARD RULE #3)
// is the acceptance gate.

#include "moss_tts_local_codec.h"

#include "core/gguf_loader.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace moss_tts_local_codec {

namespace {

constexpr float NORM_EPS = 1e-5f;
constexpr float ROPE_FREQ_BASE = 10000.0f;
constexpr int NUM_VQ = 12;
constexpr int CB_DIM = 8;
constexpr int RVQ_DIM = 512;
constexpr int OUT_DIM = 768;
constexpr int NUM_CH = 2;
constexpr int DOWNSAMPLE_RATE = 3840; // per output channel; interleaved hop = 3840*2

// Query-chunk size for the codec attention. The decoder transformers are causal
// + sliding-window (context <= 400), so a dense T x T attention is unnecessary
// and blows up (T reaches t_audio*32 = 100k+ at the final stage -> O(T^2) OOM).
// Chunking queries into QCHUNK-sized blocks, each attending only its windowed
// keys, bounds peak memory to O(QCHUNK * (QCHUNK + context)) with byte-identical
// output (this is the HF reference's query_chunk_size=1500 path). QCHUNK must be
// > the max context (400).
constexpr int QCHUNK = 2048;

struct StageSpec {
    int input_dim;
    int d_model;
    int n_heads;
    int dim_ff;
    int n_layers;
    int output_dim;
    int patch_after; // upsample factor applied AFTER this stage
    int gguf_idx;    // codec.dec.<this>
    int context;     // sliding-window causal attention span, in keys
};

// Decoder stack (decoder_kwargs): 12.5Hz -> 96kHz interleaved (see STUDY-4B.md).
// Contexts = round(frame_rate * context_duration): 12.5*10, 25*10, 50*8, 100*4,
// 200*2, 400*1 = 125, 250, 400, 400, 400, 400.
constexpr std::array<StageSpec, 6> DECODER_STAGES = {{
    {768, 1280, 20, 5120, 32, 1280, 2, 0, 125},
    {640, 768, 12, 3072, 12, 768, 2, 2, 250},
    {384, 768, 12, 3072, 12, 768, 2, 4, 400},
    {384, 768, 12, 3072, 12, 768, 2, 6, 400},
    {384, 768, 12, 3072, 12, 768, 2, 8, 400},
    {384, 768, 12, 3072, 12, 240, 240, 10, 400},
}};

} // namespace

// ===========================================================================
// Codec
// ===========================================================================

struct Codec {
    ggml_backend_t backend = nullptr;
    ggml_backend_sched_t sched = nullptr;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Reconstructed (post-weight-norm) projection weights, own buffer.
    ggml_context* w_ctx = nullptr;
    ggml_backend_buffer_t w_buf = nullptr;
    std::array<ggml_tensor*, NUM_VQ> q_oproj_w{};
    std::array<ggml_tensor*, NUM_VQ> q_oproj_b{};
    ggml_tensor* quant_oproj_w = nullptr;
    ggml_tensor* quant_oproj_b = nullptr;

    std::array<ggml_tensor*, NUM_VQ> codebook{};

    struct Layer {
        ggml_tensor *norm1_w, *norm1_b, *norm2_w, *norm2_b;
        ggml_tensor *attn_in, *attn_out, *ffn1, *ffn2;
        ggml_tensor *layer_scale_1, *layer_scale_2;
    };
    struct Stage {
        StageSpec spec;
        ggml_tensor* iproj = nullptr; // input_proj (always present)
        ggml_tensor* oproj = nullptr; // output_proj (always present)
        std::vector<Layer> layers;
    };
    std::array<Stage, 6> stages;

    std::vector<uint8_t> compute_meta;

    ggml_tensor* get(const std::string& name) const {
        auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : it->second;
    }
    ggml_tensor* req(const std::string& name) const {
        ggml_tensor* t = get(name);
        if (!t)
            fprintf(stderr, "moss_tts_local_codec: missing tensor %s\n", name.c_str());
        return t;
    }
};

namespace {

// f16/f32 device tensor -> host f32 vector.
std::vector<float> read_f32(ggml_tensor* t) {
    const size_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else { // F16
        std::vector<ggml_fp16_t> h(n);
        ggml_backend_tensor_get(t, h.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; i++)
            out[i] = ggml_fp16_to_fp32(h[i]);
    }
    return out;
}

// Reconstruct effective Conv1d(k=1) weight from weight-norm params wp0/wp1:
//   w[o,i] = wp0[o] * wp1[o,i] / sqrt(Sum_i wp1[o,i]^2)
// Uploads dst_w as f16 in (in, out) layout, dst_b as f16 (out,).
bool reconstruct(const Codec& c, const std::string& base, ggml_tensor* dst_w, ggml_tensor* dst_b, int in_dim,
                 int out_dim) {
    ggml_tensor* wp0_t = c.get(base + "wp0");
    ggml_tensor* wp1_t = c.get(base + "wp1");
    ggml_tensor* bias_t = c.get(base + "bias");
    if (!wp0_t || !wp1_t)
        return false;
    if ((int)ggml_nelements(wp0_t) != out_dim || (int)ggml_nelements(wp1_t) != in_dim * out_dim)
        return false;
    std::vector<float> wp0 = read_f32(wp0_t);
    std::vector<float> wp1 = read_f32(wp1_t);
    std::vector<ggml_fp16_t> w_eff((size_t)in_dim * out_dim);
    for (int o = 0; o < out_dim; o++) {
        float g = wp0[(size_t)o];
        float ssq = 0.f;
        for (int i = 0; i < in_dim; i++) {
            float v = wp1[(size_t)o * in_dim + i];
            ssq += v * v;
        }
        float scale = ssq > 0.f ? g / std::sqrt(ssq) : 0.f;
        for (int i = 0; i < in_dim; i++)
            w_eff[(size_t)o * in_dim + i] = ggml_fp32_to_fp16(scale * wp1[(size_t)o * in_dim + i]);
    }
    ggml_backend_tensor_set(dst_w, w_eff.data(), 0, w_eff.size() * sizeof(ggml_fp16_t));
    std::vector<ggml_fp16_t> bias((size_t)out_dim, ggml_fp32_to_fp16(0.f));
    if (bias_t) {
        std::vector<float> b = read_f32(bias_t);
        for (int o = 0; o < out_dim && o < (int)b.size(); o++)
            bias[(size_t)o] = ggml_fp32_to_fp16(b[(size_t)o]);
    }
    ggml_backend_tensor_set(dst_b, bias.data(), 0, bias.size() * sizeof(ggml_fp16_t));
    return true;
}

ggml_tensor* to_f32(ggml_context* g, ggml_tensor* t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(g, t, GGML_TYPE_F32);
}

ggml_tensor* build_layer_norm(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    ggml_tensor* y = ggml_norm(g, x, NORM_EPS);
    y = ggml_mul(g, y, to_f32(g, w));
    y = ggml_add(g, y, to_f32(g, b));
    return y;
}

// x: (d_model, T). Fused QKV (no bias) -> RoPE (adjacent-pair, base 1e4) ->
// QUERY-CHUNKED sliding-window causal SDPA (scale 1/sqrt(head_dim)) -> out proj.
// Each query block [q0,q1) attends only its windowed keys [max(0,q0-ctx+1), q1),
// so peak memory is O(QCHUNK*(QCHUNK+ctx)) not O(T^2) — numerically identical to
// a dense pass. mask_b0: (T0,T0) block-0 mask (T0=min(QCHUNK,T)); mask_int:
// ((QCHUNK+ctx-1), QCHUNK) shared interior band (null when T<=QCHUNK).
ggml_tensor* build_attention(ggml_context* g, ggml_tensor* x, const Codec::Layer& L, int d_model, int n_heads,
                             ggml_tensor* pos, int context, ggml_tensor* mask_b0, ggml_tensor* mask_int) {
    const int head_dim = d_model / n_heads;
    const int T = (int)x->ne[1];
    const float attn_scale = 1.0f / std::sqrt((float)head_dim);

    ggml_tensor* qkv = ggml_mul_mat(g, L.attn_in, x); // (3*d_model, T)
    const size_t e = ggml_type_size(qkv->type);
    const size_t row = (size_t)head_dim * e;
    ggml_tensor* Q = ggml_cont(g, ggml_view_3d(g, qkv, head_dim, n_heads, T, row, qkv->nb[1], 0));
    ggml_tensor* K = ggml_cont(g, ggml_view_3d(g, qkv, head_dim, n_heads, T, row, qkv->nb[1], (size_t)d_model * e));
    ggml_tensor* V = ggml_cont(g, ggml_view_3d(g, qkv, head_dim, n_heads, T, row, qkv->nb[1], (size_t)2 * d_model * e));

    Q = ggml_rope_ext(g, Q, pos, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, T, ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);
    K = ggml_rope_ext(g, K, pos, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, T, ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);

    Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));               // (head_dim, T, n_heads)
    K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));               // (head_dim, T, n_heads)
    ggml_tensor* Vt = ggml_cont(g, ggml_permute(g, V, 1, 2, 0, 3)); // (T, head_dim, n_heads)

    ggml_tensor* out = nullptr; // accumulated (head_dim, T, n_heads)
    for (int q0 = 0; q0 < T; q0 += QCHUNK) {
        const int q1 = q0 + QCHUNK < T ? q0 + QCHUNK : T;
        const int Qc = q1 - q0;
        const int kw0 = (q0 == 0) ? 0 : (q0 - (context - 1) > 0 ? q0 - (context - 1) : 0);
        const int Kw = q1 - kw0;

        ggml_tensor* Qb = ggml_cont(g, ggml_view_3d(g, Q, head_dim, Qc, n_heads, Q->nb[1], Q->nb[2], q0 * Q->nb[1]));
        ggml_tensor* Kb = ggml_cont(g, ggml_view_3d(g, K, head_dim, Kw, n_heads, K->nb[1], K->nb[2], kw0 * K->nb[1]));
        // Vt is (T, head_dim, n_heads); take key-rows [kw0, q1) along dim0.
        ggml_tensor* Vb =
            ggml_cont(g, ggml_view_3d(g, Vt, Kw, head_dim, n_heads, Vt->nb[1], Vt->nb[2], kw0 * Vt->nb[0]));

        ggml_tensor* sc = ggml_mul_mat(g, Kb, Qb); // (Kw, Qc, n_heads)

        ggml_tensor* m = (q0 == 0) ? mask_b0 : mask_int;
        if (Kw != (int)m->ne[0] || Qc != (int)m->ne[1]) // partial (last) block -> top-left band
            m = ggml_cont(g, ggml_view_2d(g, m, Kw, Qc, m->nb[1], 0));
        sc = ggml_soft_max_ext(g, sc, m, attn_scale, 0.0f);

        ggml_tensor* ab = ggml_mul_mat(g, Vb, sc); // (head_dim, Qc, n_heads)
        out = out ? ggml_concat(g, out, ab, 1) : ab;
    }

    out = ggml_cont(g, ggml_permute(g, out, 0, 2, 1, 3)); // (head_dim, n_heads, T)
    out = ggml_reshape_2d(g, out, d_model, T);
    out = ggml_mul_mat(g, L.attn_out, out); // (d_model, T)
    return out;
}

// Pre-norm layer: x + ls1*attn(LN1(x)); x + ls2*ffn(LN2(x)). FFN = Linear ->
// GELU(erf) -> Linear (no bias). LayerScale is per-channel (d_model).
ggml_tensor* build_layer(ggml_context* g, ggml_tensor* x, const Codec::Layer& L, int d_model, int n_heads,
                         ggml_tensor* pos, int context, ggml_tensor* mask_b0, ggml_tensor* mask_int) {
    ggml_tensor* y = build_layer_norm(g, x, L.norm1_w, L.norm1_b);
    y = build_attention(g, y, L, d_model, n_heads, pos, context, mask_b0, mask_int);
    y = ggml_mul(g, y, to_f32(g, L.layer_scale_1));
    x = ggml_add(g, x, y);

    y = build_layer_norm(g, x, L.norm2_w, L.norm2_b);
    y = ggml_mul_mat(g, L.ffn1, y);
    y = ggml_gelu_erf(g, y); // nn.GELU() default = exact erf (NOT tanh approx)
    y = ggml_mul_mat(g, L.ffn2, y);
    y = ggml_mul(g, y, to_f32(g, L.layer_scale_2));
    x = ggml_add(g, x, y);
    return x;
}

// ProjectedTransformer: input_proj -> N layers -> output_proj (both always).
ggml_tensor* build_stage(ggml_context* g, ggml_tensor* x, const Codec::Stage& S, ggml_tensor* pos, ggml_tensor* mask_b0,
                         ggml_tensor* mask_int) {
    x = ggml_mul_mat(g, S.iproj, x); // (d_model, T)
    for (const auto& L : S.layers)
        x = build_layer(g, x, L, S.spec.d_model, S.spec.n_heads, pos, S.spec.context, mask_b0, mask_int);
    x = ggml_mul_mat(g, S.oproj, x); // (output_dim, T)
    return x;
}

// PatchedPretransform.decode (upsample): (d*h, T) -> (d, T*h).
//   out[d_i, l*h + h_k] = in[d_i*h + h_k, l]   (depth-to-time)
ggml_tensor* patch_upsample(ggml_context* g, ggml_tensor* x, int patch) {
    const int64_t dh = x->ne[0];
    const int64_t T_in = x->ne[1];
    const int64_t d = dh / patch;
    ggml_tensor* y = ggml_reshape_3d(g, x, patch, d, T_in);
    y = ggml_permute(g, y, 1, 0, 2, 3); // (d, patch, T_in)
    y = ggml_cont(g, y);
    y = ggml_reshape_2d(g, y, d, T_in * patch);
    return y;
}

// Block-0 mask (T0 x T0, key-major): key kv, query q both global from 0.
// additive: 0 if kv<=q && q-kv<ctx else -inf.  flat = q*T0 + kv (ne0=key fast).
void fill_mask_b0(std::vector<float>& buf, int T0, int context) {
    buf.assign((size_t)T0 * T0, 0.f);
    for (int q = 0; q < T0; q++)
        for (int kv = 0; kv < T0; kv++)
            buf[(size_t)q * T0 + kv] = (kv <= q && (q - kv) < context) ? 0.f : -INFINITY;
}

// Interior-block band mask (Kw_max x QCHUNK, key-major). For an interior query
// block the keys start ctx-1 before the block, so key-local j and query-local i
// are valid iff j>=i && j-i<ctx (see build_attention derivation).
// Kw_max = QCHUNK + context - 1.  flat = i*Kw_max + j (ne0=key fast).
void fill_mask_interior(std::vector<float>& buf, int context) {
    const int kw_max = QCHUNK + context - 1;
    buf.assign((size_t)kw_max * QCHUNK, 0.f);
    for (int i = 0; i < QCHUNK; i++)
        for (int j = 0; j < kw_max; j++)
            buf[(size_t)i * kw_max + j] = (j >= i && (j - i) < context) ? 0.f : -INFINITY;
}

} // namespace

// ===========================================================================
// Load
// ===========================================================================

Codec* load(const char* path, ggml_backend_t backend, ggml_backend_sched_t sched, int verbosity) {
    if (!path || !backend)
        return nullptr;
    auto* c = new Codec();
    c->backend = backend;
    c->sched = sched;

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "moss_tts_local_codec", wl)) {
        fprintf(stderr, "moss_tts_local_codec: failed to load weights from %s\n", path);
        delete c;
        return nullptr;
    }
    c->ctx = wl.ctx;
    c->buf = wl.buf;
    c->tensors = std::move(wl.tensors);

    // Codebooks + decoder stage weights.
    bool ok = true;
    for (int i = 0; i < NUM_VQ; i++) {
        c->codebook[i] = c->req("codec.quant.q." + std::to_string(i) + ".codebook");
        ok = ok && c->codebook[i];
    }
    for (size_t s = 0; s < c->stages.size(); s++) {
        auto& S = c->stages[s];
        S.spec = DECODER_STAGES[s];
        const std::string base = "codec.dec." + std::to_string(S.spec.gguf_idx) + ".";
        S.iproj = c->req(base + "iproj.weight");
        S.oproj = c->req(base + "oproj.weight");
        ok = ok && S.iproj && S.oproj;
        S.layers.resize((size_t)S.spec.n_layers);
        for (int li = 0; li < S.spec.n_layers; li++) {
            const std::string lb = base + "l." + std::to_string(li) + ".";
            auto& L = S.layers[(size_t)li];
            L.norm1_w = c->req(lb + "norm1.weight");
            L.norm1_b = c->req(lb + "norm1.bias");
            L.norm2_w = c->req(lb + "norm2.weight");
            L.norm2_b = c->req(lb + "norm2.bias");
            L.attn_in = c->req(lb + "attn_in.weight");
            L.attn_out = c->req(lb + "attn_out.weight");
            L.ffn1 = c->req(lb + "ffn1.weight");
            L.ffn2 = c->req(lb + "ffn2.weight");
            L.layer_scale_1 = c->req(lb + "ls1.scale");
            L.layer_scale_2 = c->req(lb + "ls2.scale");
            ok = ok && L.norm1_w && L.attn_in && L.ffn1 && L.layer_scale_1;
        }
    }
    if (!ok) {
        fprintf(stderr, "moss_tts_local_codec: missing decoder tensors\n");
        free(c);
        return nullptr;
    }

    // Reconstruct weight-normed quantizer projections into a dedicated buffer.
    {
        ggml_init_params ip = {ggml_tensor_overhead() * (2 * NUM_VQ + 8), nullptr, true};
        c->w_ctx = ggml_init(ip);
        auto mk_w = [&](int in, int out) { return ggml_new_tensor_2d(c->w_ctx, GGML_TYPE_F16, in, out); };
        auto mk_b = [&](int out) { return ggml_new_tensor_1d(c->w_ctx, GGML_TYPE_F16, out); };
        for (int i = 0; i < NUM_VQ; i++) {
            c->q_oproj_w[i] = mk_w(CB_DIM, RVQ_DIM);
            c->q_oproj_b[i] = mk_b(RVQ_DIM);
        }
        c->quant_oproj_w = mk_w(RVQ_DIM, OUT_DIM);
        c->quant_oproj_b = mk_b(OUT_DIM);
        c->w_buf = ggml_backend_alloc_ctx_tensors(c->w_ctx, backend);
        if (!c->w_buf) {
            fprintf(stderr, "moss_tts_local_codec: alloc weight buffer failed\n");
            free(c);
            return nullptr;
        }
        bool rok = true;
        for (int i = 0; i < NUM_VQ; i++) {
            const std::string b = "codec.quant.q." + std::to_string(i) + ".oproj.";
            rok = rok && reconstruct(*c, b, c->q_oproj_w[i], c->q_oproj_b[i], CB_DIM, RVQ_DIM);
        }
        rok = rok && reconstruct(*c, "codec.quant.oproj.", c->quant_oproj_w, c->quant_oproj_b, RVQ_DIM, OUT_DIM);
        if (!rok) {
            fprintf(stderr, "moss_tts_local_codec: weight-norm reconstruction failed\n");
            free(c);
            return nullptr;
        }
    }

    // Query-chunked attention emits many small blocks for long audio; size the
    // graph generously (a 4096-frame decode is ~40k nodes).
    c->compute_meta.resize(ggml_tensor_overhead() * 262144 + ggml_graph_overhead_custom(262144, false));
    if (verbosity >= 1)
        fprintf(stderr, "moss_tts_local_codec: loaded %s (%zu tensors)\n", path, c->tensors.size());
    return c;
}

void free(Codec* c) {
    if (!c)
        return;
    if (c->w_buf)
        ggml_backend_buffer_free(c->w_buf);
    if (c->w_ctx)
        ggml_free(c->w_ctx);
    if (c->buf)
        ggml_backend_buffer_free(c->buf);
    if (c->ctx)
        ggml_free(c->ctx);
    delete c;
}

int interleaved_hop(const Codec* c) {
    (void)c;
    return DOWNSAMPLE_RATE * NUM_CH;
}
int num_channels(const Codec* c) {
    (void)c;
    return NUM_CH;
}
int sampling_rate(const Codec* c) {
    (void)c;
    return 48000;
}

// ===========================================================================
// Decode
// ===========================================================================

std::vector<float> decode(Codec* c, const int32_t* codes, int n_vq, int t_audio) {
    if (!c || !codes || t_audio <= 0 || n_vq < 1 || n_vq > NUM_VQ)
        return {};
    const int64_t n_samples = (int64_t)t_audio * DOWNSAMPLE_RATE * NUM_CH;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* g = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(g, 262144, false);

    // Inputs: per-quantizer code vectors + per-stage positions + masks.
    std::vector<ggml_tensor*> codes_in(n_vq);
    for (int i = 0; i < n_vq; i++) {
        codes_in[i] = ggml_new_tensor_1d(g, GGML_TYPE_I32, t_audio);
        char nm[32];
        snprintf(nm, sizeof(nm), "codes_%d", i);
        ggml_set_name(codes_in[i], nm);
        ggml_set_input(codes_in[i]);
    }
    int T_at[6];
    T_at[0] = t_audio;
    for (int s = 1; s < 6; s++)
        T_at[s] = T_at[s - 1] * DECODER_STAGES[s - 1].patch_after;
    std::array<ggml_tensor*, 6> pos_T{}, mask_b0{}, mask_int{};
    for (int s = 0; s < 6; s++) {
        pos_T[s] = ggml_new_tensor_1d(g, GGML_TYPE_I32, T_at[s]);
        char nm[16];
        snprintf(nm, sizeof(nm), "pos_%d", s);
        ggml_set_name(pos_T[s], nm);
        ggml_set_input(pos_T[s]);
        const int T0 = T_at[s] < QCHUNK ? T_at[s] : QCHUNK;
        mask_b0[s] = ggml_new_tensor_2d(g, GGML_TYPE_F32, T0, T0);
        snprintf(nm, sizeof(nm), "mb0_%d", s);
        ggml_set_name(mask_b0[s], nm);
        ggml_set_input(mask_b0[s]);
        if (T_at[s] > QCHUNK) {
            const int kwm = QCHUNK + DECODER_STAGES[s].context - 1;
            mask_int[s] = ggml_new_tensor_2d(g, GGML_TYPE_F32, kwm, QCHUNK);
            snprintf(nm, sizeof(nm), "mint_%d", s);
            ggml_set_name(mask_int[s], nm);
            ggml_set_input(mask_int[s]);
        } else {
            mask_int[s] = nullptr;
        }
    }

    // Quantizer.decode_codes: Sum_i q.oproj(codebook[i][codes_i]) -> global oproj.
    ggml_tensor* sum = nullptr;
    for (int i = 0; i < n_vq; i++) {
        ggml_tensor* z = ggml_get_rows(g, c->codebook[i], codes_in[i]); // (8, T)
        z = ggml_mul_mat(g, c->q_oproj_w[i], z);                        // (512, T)
        z = ggml_add(g, z, to_f32(g, c->q_oproj_b[i]));
        sum = sum ? ggml_add(g, sum, z) : z;
    }
    ggml_tensor* x = ggml_mul_mat(g, c->quant_oproj_w, sum); // (768, T)
    x = ggml_add(g, x, to_f32(g, c->quant_oproj_b));

    for (int s = 0; s < 6; s++) {
        x = build_stage(g, x, c->stages[s], pos_T[s], mask_b0[s], mask_int[s]);
        x = patch_upsample(g, x, c->stages[s].spec.patch_after);
    }
    // x is now (1, t_audio * 7680) mono channel-interleaved.

    ggml_tensor* waveform = ggml_reshape_1d(g, x, n_samples);
    ggml_set_name(waveform, "waveform");
    ggml_set_output(waveform);
    ggml_build_forward_expand(gf, waveform);

    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "moss_tts_local_codec: alloc decode graph failed\n");
        ggml_free(g);
        return {};
    }

    for (int i = 0; i < n_vq; i++) {
        std::vector<int32_t> col(t_audio);
        for (int t = 0; t < t_audio; t++)
            col[(size_t)t] = codes[(size_t)i * t_audio + t];
        ggml_backend_tensor_set(codes_in[i], col.data(), 0, (size_t)t_audio * sizeof(int32_t));
    }
    for (int s = 0; s < 6; s++) {
        std::vector<int32_t> p((size_t)T_at[s]);
        for (int t = 0; t < T_at[s]; t++)
            p[(size_t)t] = t;
        ggml_backend_tensor_set(pos_T[s], p.data(), 0, p.size() * sizeof(int32_t));
        const int ctx = DECODER_STAGES[s].context;
        const int T0 = T_at[s] < QCHUNK ? T_at[s] : QCHUNK;
        std::vector<float> b0;
        fill_mask_b0(b0, T0, ctx);
        ggml_backend_tensor_set(mask_b0[s], b0.data(), 0, b0.size() * sizeof(float));
        if (mask_int[s]) {
            std::vector<float> mi;
            fill_mask_interior(mi, ctx);
            ggml_backend_tensor_set(mask_int[s], mi.data(), 0, mi.size() * sizeof(float));
        }
    }

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "moss_tts_local_codec: decode compute failed\n");
        ggml_free(g);
        return {};
    }

    std::vector<float> wav((size_t)n_samples);
    ggml_tensor* out = ggml_graph_get_tensor(gf, "waveform");
    ggml_backend_tensor_get(out, wav.data(), 0, wav.size() * sizeof(float));
    ggml_free(g);
    return wav;
}

} // namespace moss_tts_local_codec
