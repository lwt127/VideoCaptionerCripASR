// Regression coverage for CrispASR's carried ggml audio-op backend patches.
//
// This intentionally runs GGML_OP_CONV_TRANSPOSE_1D and GGML_OP_COL2IM_1D
// directly on one selected GPU backend, with no CPU fallback scheduler. If a
// ggml bump drops the backend kernels or dtype support, the test fails at graph
// compute time instead of silently passing through CPU.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#if defined(CRISPASR_TEST_BACKEND_METAL)
#include "ggml-metal.h"
#elif defined(CRISPASR_TEST_BACKEND_VULKAN)
#include "ggml-vulkan.h"
#elif defined(CRISPASR_TEST_BACKEND_CUDA)
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static float value_at(int i, int salt) {
    int v = (i * 37 + salt * 53) % 211;
    return ((float)v - 105.0f) / 37.0f;
}

static ggml_backend_t init_test_backend() {
#if defined(CRISPASR_TEST_BACKEND_METAL)
    return ggml_backend_metal_init();
#elif defined(CRISPASR_TEST_BACKEND_VULKAN)
    if (ggml_backend_vk_get_device_count() <= 0)
        return nullptr;
    return ggml_backend_vk_init(0);
#elif defined(CRISPASR_TEST_BACKEND_CUDA)
    if (ggml_backend_cuda_get_device_count() <= 0)
        return nullptr;
    return ggml_backend_cuda_init(0);
#else
    return nullptr;
#endif
}

static const char * backend_name() {
#if defined(CRISPASR_TEST_BACKEND_METAL)
    return "metal";
#elif defined(CRISPASR_TEST_BACKEND_VULKAN)
    return "vulkan";
#elif defined(CRISPASR_TEST_BACKEND_CUDA)
    return "cuda";
#else
    return "unknown";
#endif
}

static bool close_enough(const std::vector<float> & a, const std::vector<float> & b, float tol, const char * tag) {
    if (a.size() != b.size()) {
        std::fprintf(stderr, "%s: size mismatch %zu vs %zu\n", tag, a.size(), b.size());
        return false;
    }
    float max_abs = 0.0f;
    double mse = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        float d = std::fabs(a[i] - b[i]);
        max_abs = std::max(max_abs, d);
        mse += (double)d * d;
    }
    double rmse = std::sqrt(mse / std::max<size_t>(a.size(), 1));
    std::printf("%s: max_abs=%g rmse=%g n=%zu\n", tag, max_abs, rmse, a.size());
    return max_abs <= tol;
}

static bool run_conv_transpose_1d(ggml_backend_t backend, std::vector<float> & out) {
    const int K = 5;
    const int Cout = 4;
    const int Cin = 3;
    const int T = 9;
    const int stride = 3;

    std::vector<ggml_fp16_t> w((size_t)K * Cout * Cin);
    std::vector<float> x((size_t)T * Cin);
    for (size_t i = 0; i < w.size(); i++)
        w[i] = ggml_fp32_to_fp16(value_at((int)i, 1));
    for (size_t i = 0; i < x.size(); i++)
        x[i] = value_at((int)i, 2);

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * wt = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K, Cout, Cin);
    ggml_tensor * xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, Cin);
    ggml_set_name(wt, "w");
    ggml_set_name(xt, "x");
    ggml_set_input(wt);
    ggml_set_input(xt);

    ggml_tensor * y = ggml_conv_transpose_1d(ctx, wt, xt, stride, 0, 1);
    ggml_set_name(y, "y");
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::fprintf(stderr, "%s conv_transpose_1d: alloc_ctx_tensors failed\n", backend_name());
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(wt, w.data(), 0, w.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(xt, x.data(), 0, x.size() * sizeof(float));
    enum ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s conv_transpose_1d: compute failed (%d)\n", backend_name(), (int)st);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    out.assign((size_t)ggml_nelements(y), 0.0f);
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

static std::vector<float> col2im_reference(const std::vector<float> & col, int T_in, int T_out,
                                           int OC, int K, int s0, int p0) {
    const int K_OC = K * OC;
    std::vector<float> out((size_t)T_out * OC, 0.0f);
    for (int idx = 0; idx < T_out * OC; idx++) {
        const int t_out = idx % T_out;
        const int oc = idx / T_out;
        const int t_abs = t_out + p0;
        int t_in_min = (t_abs - K + s0) / s0;
        if (t_in_min < 0)
            t_in_min = 0;
        int t_in_max = t_abs / s0;
        if (t_in_max >= T_in)
            t_in_max = T_in - 1;
        float sum = 0.0f;
        for (int t_in = t_in_min; t_in <= t_in_max; t_in++) {
            const int k = t_abs - t_in * s0;
            sum += col[(size_t)(oc * K + k) + (size_t)t_in * K_OC];
        }
        out[idx] = sum;
    }
    return out;
}

static bool run_col2im_1d(ggml_backend_t backend, ggml_type type, std::vector<float> & out,
                          std::vector<float> & expected) {
    const int K = 4;
    const int OC = 3;
    const int T_in = 8;
    const int stride = 2;
    const int crop_left = 1;
    const int T_out = (T_in - 1) * stride + K - crop_left;
    const int K_OC = K * OC;

    std::vector<float> col_f32((size_t)K_OC * T_in);
    for (size_t i = 0; i < col_f32.size(); i++)
        col_f32[i] = value_at((int)i, 3);
    expected = col2im_reference(col_f32, T_in, T_out, OC, K, stride, crop_left);

    std::vector<ggml_fp16_t> col_f16;
    if (type == GGML_TYPE_F16) {
        col_f16.resize(col_f32.size());
        for (size_t i = 0; i < col_f32.size(); i++) {
            col_f16[i] = ggml_fp32_to_fp16(col_f32[i]);
            col_f32[i] = ggml_fp16_to_fp32(col_f16[i]);
        }
        expected = col2im_reference(col_f32, T_in, T_out, OC, K, stride, crop_left);
    }

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * col = ggml_new_tensor_2d(ctx, type, K_OC, T_in);
    ggml_set_name(col, "col");
    ggml_set_input(col);
    ggml_tensor * y = ggml_col2im_1d(ctx, col, stride, OC, crop_left);
    ggml_set_name(y, "y");
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::fprintf(stderr, "%s col2im_1d: alloc_ctx_tensors failed\n", backend_name());
        ggml_free(ctx);
        return false;
    }

    if (type == GGML_TYPE_F16) {
        ggml_backend_tensor_set(col, col_f16.data(), 0, col_f16.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(col, col_f32.data(), 0, col_f32.size() * sizeof(float));
    }
    enum ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s col2im_1d: compute failed (%d)\n", backend_name(), (int)st);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    out.assign((size_t)ggml_nelements(y), 0.0f);
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t gpu = init_test_backend();
    if (!gpu) {
        std::fprintf(stderr, "%s backend unavailable at runtime; skipping\n", backend_name());
        ggml_backend_free(cpu);
        return 4;
    }

    std::vector<float> conv_cpu;
    std::vector<float> conv_gpu;
    bool ok = run_conv_transpose_1d(cpu, conv_cpu) &&
              run_conv_transpose_1d(gpu, conv_gpu) &&
              close_enough(conv_cpu, conv_gpu, 5e-3f, "conv_transpose_1d_f16_weight");

    std::vector<float> col_gpu;
    std::vector<float> col_ref;
    ok = ok && run_col2im_1d(gpu, GGML_TYPE_F32, col_gpu, col_ref) &&
         close_enough(col_ref, col_gpu, 1e-5f, "col2im_1d_f32");

#if defined(CRISPASR_TEST_COL2IM_F16)
    ok = ok && run_col2im_1d(gpu, GGML_TYPE_F16, col_gpu, col_ref) &&
         close_enough(col_ref, col_gpu, 2e-3f, "col2im_1d_f16");
#endif

    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return ok ? 0 : 1;
}
