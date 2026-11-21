#include "convert.cuh"
#include "inkling2.cuh"

// lanes that share one output row in the up kernel
static constexpr int INKLING2_CTRL_UP_LANES = 8;

// load v values starting at p as float; v > 1 loads 16-byte chunks, p must be 16-byte aligned
template <int v, typename T>
static __device__ __forceinline__ void inkling2_load(const T * p, float * dst) {
    if constexpr (v == 1) {
        dst[0] = ggml_cuda_cast<float>(p[0]);
    } else {
        static_assert(v*sizeof(T) % 16 == 0, "bad chunk size");
        T tmp[v];
#pragma unroll
        for (int q = 0; q < int(v*sizeof(T)/16); ++q) {
            ((int4 *) tmp)[q] = ((const int4 *) p)[q];
        }
#pragma unroll
        for (int k = 0; k < v; ++k) {
            dst[k] = ggml_cuda_cast<float>(tmp[k]);
        }
    }
}

// g[i, t] = down[i, :n].cur[t]/rms(cur[t]) + down[i, n:].msg[t]/rms(msg[t]), one block per (i, t)
template <typename T, bool vec>
static __global__ void inkling2_ctrl_down(
        const float * __restrict__ cur, const float * __restrict__ msg, const T * __restrict__ down, float * __restrict__ g,
        const int n, const int r, const int64_t s_cur, const int64_t s_msg, const int64_t s_down, const float eps) {
    constexpr int v = vec ? 16/sizeof(T) : 1;

    const int i = blockIdx.x;
    const int t = blockIdx.y;

    const float * c = cur  + t*s_cur;
    const float * m = msg  + t*s_msg;
    const T     * w = down + i*s_down;

    float2 ss  = make_float2(0.0f, 0.0f);
    float2 dot = make_float2(0.0f, 0.0f);

    for (int k = threadIdx.x*v; k < n; k += blockDim.x*v) {
        float wc[v];
        float wm[v];
        float xc[v];
        float xm[v];
        inkling2_load<v>(w + k,     wc);
        inkling2_load<v>(w + n + k, wm);
        inkling2_load<v>(c + k,     xc);
        inkling2_load<v>(m + k,     xm);
#pragma unroll
        for (int q = 0; q < v; ++q) {
            ss.x  += xc[q]*xc[q];
            ss.y  += xm[q]*xm[q];
            dot.x += wc[q]*xc[q];
            dot.y += wm[q]*xm[q];
        }
    }

    __shared__ float2 buf_ss[32];
    __shared__ float2 buf_dot[32];
    ss  = block_reduce<block_reduce_method::SUM>(ss,  buf_ss);
    dot = block_reduce<block_reduce_method::SUM>(dot, buf_dot);

    if (threadIdx.x == 0) {
        g[t*r + i] = dot.x*rsqrtf(ss.x/n + eps) + dot.y*rsqrtf(ss.y/n + eps);
    }
}

// dst[j, t] = cur - cur*tanh(forget[j].g[t]) + msg*tanh(gain[j].g[t]), INKLING2_CTRL_UP_LANES lanes per (j, t)
template <typename T, bool vec>
static __global__ void inkling2_ctrl_up(
        const float * __restrict__ cur, const float * __restrict__ msg, const T * __restrict__ gain, const T * __restrict__ forget,
        const float * __restrict__ g, float * __restrict__ dst, const int n, const int r,
        const int64_t s_cur, const int64_t s_msg, const int64_t s_gain, const int64_t s_forget, const int64_t s_dst) {
    constexpr int v     = vec ? 16/sizeof(T) : 1;
    constexpr int lanes = INKLING2_CTRL_UP_LANES;

    extern __shared__ float g_s[];

    const int t = blockIdx.y;
    for (int k = threadIdx.x; k < r; k += blockDim.x) {
        g_s[k] = g[t*r + k];
    }
    __syncthreads();

    const int  j     = blockIdx.x*(blockDim.x/lanes) + threadIdx.x/lanes;
    const int  lane  = threadIdx.x % lanes;
    const bool valid = j < n;

    float f = 0.0f;
    float a = 0.0f;
    if (valid) {
        for (int k = lane*v; k < r; k += lanes*v) {
            float wf[v];
            float wa[v];
            inkling2_load<v>(forget + j*s_forget + k, wf);
            inkling2_load<v>(gain   + j*s_gain   + k, wa);
#pragma unroll
            for (int q = 0; q < v; ++q) {
                f += wf[q]*g_s[k + q];
                a += wa[q]*g_s[k + q];
            }
        }
    }
    f = warp_reduce_sum<lanes>(f);
    a = warp_reduce_sum<lanes>(a);

    if (valid && lane == 0) {
        const float c = cur[t*s_cur + j];
        const float m = msg[t*s_msg + j];
        dst[t*s_dst + j] = (c - c*tanhf(f)) + m*tanhf(a);
    }
}

static bool inkling2_aligned(const void * p, size_t stride) {
    return (uintptr_t) p % 16 == 0 && stride % 16 == 0;
}

template <typename T>
static void inkling2_ctrl_down_cuda(const ggml_tensor * cur, const ggml_tensor * msg, const ggml_tensor * down, float * g, float eps, cudaStream_t stream) {
    const int     n  = cur->ne[0];
    const int     r  = down->ne[1];
    const int64_t nt = cur->ne[1];

    const bool vec = n % (16/sizeof(T)) == 0 &&
        inkling2_aligned(cur->data, cur->nb[1]) && inkling2_aligned(msg->data, msg->nb[1]) && inkling2_aligned(down->data, down->nb[1]);

    const dim3 blocks(r, nt);
    const dim3 threads(256);
    auto kernel = vec ? inkling2_ctrl_down<T, true> : inkling2_ctrl_down<T, false>;
    kernel<<<blocks, threads, 0, stream>>>((const float *) cur->data, (const float *) msg->data, (const T *) down->data, g,
            n, r, cur->nb[1]/sizeof(float), msg->nb[1]/sizeof(float), down->nb[1]/sizeof(T), eps);
}

template <typename T>
static void inkling2_ctrl_up_cuda(const ggml_tensor * cur, const ggml_tensor * msg, const ggml_tensor * gain, const ggml_tensor * forget,
        const float * g, ggml_tensor * dst, cudaStream_t stream) {
    const int     n  = cur->ne[0];
    const int     r  = gain->ne[0];
    const int64_t nt = cur->ne[1];

    const bool vec = r % (16/sizeof(T)) == 0 && inkling2_aligned(gain->data, gain->nb[1]) && inkling2_aligned(forget->data, forget->nb[1]);

    const int  n_threads = 256;
    const dim3 blocks((n + n_threads/INKLING2_CTRL_UP_LANES - 1)/(n_threads/INKLING2_CTRL_UP_LANES), nt);
    auto kernel = vec ? inkling2_ctrl_up<T, true> : inkling2_ctrl_up<T, false>;
    kernel<<<blocks, n_threads, r*sizeof(float), stream>>>((const float *) cur->data, (const float *) msg->data,
            (const T *) gain->data, (const T *) forget->data, g, (float *) dst->data, n, r,
            cur->nb[1]/sizeof(float), msg->nb[1]/sizeof(float), gain->nb[1]/sizeof(T), forget->nb[1]/sizeof(T), dst->nb[1]/sizeof(float));
}

bool ggml_cuda_inkling2_ctrl_supported(const ggml_tensor * op) {
    const ggml_tensor * down = op->src[2];
    const ggml_tensor * gain = op->src[3];

    auto type_ok = [](const ggml_tensor * w) {
        return (w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_BF16) && w->nb[0] == ggml_type_size(w->type);
    };

    return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && op->ne[1] <= 65535 &&
        type_ok(down) && type_ok(gain) && op->src[4]->type == gain->type && op->src[4]->nb[0] == gain->nb[0];
}

void ggml_cuda_op_inkling2_ctrl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cur    = dst->src[0];
    const ggml_tensor * msg    = dst->src[1];
    const ggml_tensor * down   = dst->src[2];
    const ggml_tensor * gain   = dst->src[3];
    const ggml_tensor * forget = dst->src[4];

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float> g(ctx.pool(), down->ne[1]*cur->ne[1]);

    switch (down->type) {
        case GGML_TYPE_F32:  inkling2_ctrl_down_cuda<float>      (cur, msg, down, g.get(), eps, stream); break;
        case GGML_TYPE_F16:  inkling2_ctrl_down_cuda<half>       (cur, msg, down, g.get(), eps, stream); break;
        case GGML_TYPE_BF16: inkling2_ctrl_down_cuda<nv_bfloat16>(cur, msg, down, g.get(), eps, stream); break;
        default: GGML_ABORT("unsupported type %s", ggml_type_name(down->type));
    }

    switch (gain->type) {
        case GGML_TYPE_F32:  inkling2_ctrl_up_cuda<float>      (cur, msg, gain, forget, g.get(), dst, stream); break;
        case GGML_TYPE_F16:  inkling2_ctrl_up_cuda<half>       (cur, msg, gain, forget, g.get(), dst, stream); break;
        case GGML_TYPE_BF16: inkling2_ctrl_up_cuda<nv_bfloat16>(cur, msg, gain, forget, g.get(), dst, stream); break;
        default: GGML_ABORT("unsupported type %s", ggml_type_name(gain->type));
    }
}
