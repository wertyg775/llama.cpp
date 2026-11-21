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

// dst = y/max(1, (|y| + eps)/threshold)*tau with y = x*scale, one warp per row
template <bool has_scale, bool has_tau>
static __global__ void inkling2_clip_f32(
        const float * __restrict__ x, const float * __restrict__ scale, const float * __restrict__ tau, float * __restrict__ dst,
        const int ne00, const int ne01, const int ne02, const int64_t nr, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s_scale1, const int s_tau2, const float inv_threshold, const float eps_threshold) {
    const int64_t ir = (int64_t) blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    if (ir >= nr) {
        return;
    }
    const int lane = threadIdx.x % WARP_SIZE;

    const int64_t i1 = ir % ne01;
    const int64_t i2 = (ir/ne01) % ne02;
    const int64_t i3 = ir/((int64_t) ne01*ne02);

    const float * xs = x + i1*s01 + i2*s02 + i3*s03;
    const float * ss = has_scale ? scale + i1*s_scale1 : nullptr;
    float       * y  = dst + ir*ne00;

    float sum = 0.0f;
    for (int i0 = lane; i0 < ne00; i0 += WARP_SIZE) {
        const float v = has_scale ? xs[i0]*ss[i0] : xs[i0];
        sum += v*v;
    }
    sum = warp_reduce_sum(sum);

    const float d = fmaxf(sqrtf(sum)*inv_threshold + eps_threshold, 1.0f);
    const float t = has_tau ? tau[i2*s_tau2] : 1.0f;

    for (int i0 = lane; i0 < ne00; i0 += WARP_SIZE) {
        const float v = has_scale ? xs[i0]*ss[i0] : xs[i0];
        y[i0] = v/d*t;
    }
}

void ggml_cuda_op_inkling2_clip(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x     = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * tau   = dst->src[2];

    GGML_ASSERT(ggml_is_contiguous(dst));

    float threshold;
    float eps;
    memcpy(&threshold, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&eps,       (const float *) dst->op_params + 1, sizeof(float));

    const int64_t nr = ggml_nrows(x);

    const int  rows_per_block = 256/WARP_SIZE;
    const dim3 blocks((nr + rows_per_block - 1)/rows_per_block);
    const dim3 threads(rows_per_block*WARP_SIZE);

    const int64_t s_scale1 = scale && scale->ne[1] > 1 ? x->ne[0] : 0;
    const int     s_tau2   = tau && tau->ne[2] > 1 ? 1 : 0;

    auto kernel = scale ? (tau ? inkling2_clip_f32<true,  true> : inkling2_clip_f32<true,  false>)
                        : (tau ? inkling2_clip_f32<false, true> : inkling2_clip_f32<false, false>);
    kernel<<<blocks, threads, 0, ctx.stream()>>>((const float *) x->data, scale ? (const float *) scale->data : nullptr,
            tau ? (const float *) tau->data : nullptr, (float *) dst->data, x->ne[0], x->ne[1], x->ne[2], nr,
            x->nb[1]/sizeof(float), x->nb[2]/sizeof(float), x->nb[3]/sizeof(float), s_scale1, s_tau2, 1.0f/threshold, eps/threshold);
}

// tokens of one sequence per block in the conv kernel
static constexpr int INKLING2_CONV_CHUNK = 64;

// one thread per channel c of sequence blockIdx.y, tokens of chunk blockIdx.z; the last chunk also writes the new state
template <bool has_dynamic, bool has_residual>
static __global__ void inkling2_conv_f32(
        const float * __restrict__ cur, const float * __restrict__ past, const float * __restrict__ weight, const float * __restrict__ dynamic,
        const float * __restrict__ residual, float * __restrict__ y, float * __restrict__ state, const int C, const int K, const int n_st, const int n_cph,
        const int64_t s_cur1, const int64_t s_past0, const int64_t s_past1, const int64_t s_past2, const int64_t s_dyn1, const int64_t s_res1,
        const float scale) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= C) {
        return;
    }
    const int q  = blockIdx.y;
    const int i0 = blockIdx.z*INKLING2_CONV_CHUNK;
    const int i1 = min(i0 + INKLING2_CONV_CHUNK, n_st);

    const float * xp = past + c*s_past1 + q*s_past2;
    const float * xc = cur  + c + (int64_t) q*n_st*s_cur1;
    const float * w  = weight + (int64_t) c*K;

    auto x = [&](int k) {
        return k < K - 1 ? xp[k*s_past0] : xc[(k - (K - 1))*s_cur1];
    };

    for (int i = i0; i < i1; ++i) {
        const int64_t t = (int64_t) q*n_st + i;

        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += x(i + k)*w[k];
        }

        if constexpr (has_dynamic) {
            const float * taps = dynamic + t*s_dyn1 + (c/n_cph)*K;
            for (int lag = 0; lag < K; ++lag) {
                sum += x(i + K - 1 - lag)*taps[lag];
            }
        }

        // no fused multiply-add: the product rounds first, as in the scale and add ops
        const float out = __fmul_rn(sum + x(i + K - 1), scale);
        y[t*C + c] = has_residual ? residual[t*s_res1 + c] + out : out;
    }

    if (i1 == n_st) {
        for (int k = 0; k < K - 1; ++k) {
            state[((int64_t) q*C + c)*(K - 1) + k] = x(n_st + k);
        }
    }
}

void ggml_cuda_op_inkling2_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cur     = dst->src[0];
    const ggml_tensor * past    = dst->src[1];
    const ggml_tensor * weight  = dst->src[2];
    const ggml_tensor * dynamic  = dst->src[3];
    const ggml_tensor * residual = dst->src[4];

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    const int C      = cur->ne[0];
    const int T      = cur->ne[1];
    const int K      = weight->ne[0];
    const int n_seqs = past->ne[2];
    const int n_st   = T/n_seqs;
    const int n_cph  = dynamic ? C/(dynamic->ne[0]/K) : 1;

    float * y = (float *) dst->data;

    const int  n_threads = 256;
    const dim3 blocks((C + n_threads - 1)/n_threads, n_seqs, (n_st + INKLING2_CONV_CHUNK - 1)/INKLING2_CONV_CHUNK);

    auto kernel = dynamic ? (residual ? inkling2_conv_f32<true,  true> : inkling2_conv_f32<true,  false>)
                          : (residual ? inkling2_conv_f32<false, true> : inkling2_conv_f32<false, false>);
    kernel<<<blocks, n_threads, 0, ctx.stream()>>>((const float *) cur->data, (const float *) past->data, (const float *) weight->data,
            dynamic ? (const float *) dynamic->data : nullptr, residual ? (const float *) residual->data : nullptr, y, y + (int64_t) C*T, C, K, n_st, n_cph,
            cur->nb[1]/sizeof(float), past->nb[0]/sizeof(float), past->nb[1]/sizeof(float), past->nb[2]/sizeof(float),
            dynamic ? dynamic->nb[1]/sizeof(float) : 0, residual ? residual->nb[1]/sizeof(float) : 0, scale);
}

// the most blocks a mix kernel keeps in registers
static constexpr int INKLING2_MIX_MAX_BLOCKS = 15;

// one block per (query, token): the scores of all states in one pass, the softmax, then the weighted sum
template <typename T>
static __global__ void inkling2_mix_f32(
        const float * __restrict__ x, const T * __restrict__ query, const float * __restrict__ blocks, float * __restrict__ dst,
        const int n, const int nq, const int n_blocks, const int64_t s_x1, const int64_t s_q1, const int64_t s_b1, const int64_t s_b2,
        const float scale, const float eps) {
    constexpr int nb_max = INKLING2_MIX_MAX_BLOCKS;

    const int iq = blockIdx.x;
    const int t  = blockIdx.y;

    const float * xs = x + t*s_x1;
    const T     * qs = query + iq*s_q1;
    const float * bs = blocks + t*s_b2;

    float dot[nb_max];
#pragma unroll
    for (int j = 0; j < nb_max; ++j) {
        dot[j] = 0.0f;
    }
    float ss = 0.0f;
    float dx = 0.0f;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float qi = ggml_cuda_cast<float>(qs[i]);
        const float xi = xs[i];
        ss += xi*xi;
        dx += xi*qi;
#pragma unroll
        for (int j = 0; j < nb_max; ++j) {
            if (j < n_blocks) {
                dot[j] += bs[j*s_b1 + i]*qi;
            }
        }
    }

    // per warp sums: the block dots, then the sum of squares and the dot of x
    __shared__ float red[nb_max + 2][32];
    __shared__ float p[nb_max + 2];

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int nw   = blockDim.x / WARP_SIZE;

#pragma unroll
    for (int j = 0; j < nb_max; ++j) {
        if (j < n_blocks) {
            const float v = warp_reduce_sum(dot[j]);
            if (lane == 0) {
                red[j][warp] = v;
            }
        }
    }
    ss = warp_reduce_sum(ss);
    dx = warp_reduce_sum(dx);
    if (lane == 0) {
        red[nb_max + 0][warp] = ss;
        red[nb_max + 1][warp] = dx;
    }
    __syncthreads();

    // warp 0, lane j: the score of state j (block j, then x at n_blocks), softmax in registers
    // p[j] = softmax weight of state j, p[nb_max + 1] = norm scale of x
    if (warp == 0) {
        float sum_ss = 0.0f;
        float sum_dx = 0.0f;
        float d      = 0.0f;
        for (int w = 0; w < nw; ++w) {
            sum_ss += red[nb_max + 0][w];
            sum_dx += red[nb_max + 1][w];
            if (lane < n_blocks) {
                d += red[lane][w];
            }
        }
        const float s = rsqrtf(sum_ss/n + eps);
        if (lane == n_blocks) {
            d = sum_dx*s;
        }

        const float sc  = lane <= n_blocks ? d*scale : -INFINITY;
        const float max = warp_reduce_max(sc);
        const float e   = lane <= n_blocks ? expf(sc - max) : 0.0f;
        const float sum = warp_reduce_sum(e);
        if (lane <= n_blocks) {
            p[lane] = e/sum;
        }
        if (lane == 0) {
            p[nb_max + 1] = s;
        }
    }
    __syncthreads();

    const float px = p[n_blocks];
    const float s  = p[nb_max + 1];

    float * y = dst + ((int64_t) t*nq + iq)*n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < n_blocks; ++j) {
            acc += bs[j*s_b1 + i]*p[j];
        }
        y[i] = acc + (xs[i]*s)*px;
    }
}

bool ggml_cuda_inkling2_mix_supported(const ggml_tensor * op) {
    const ggml_tensor * query  = op->src[1];
    const ggml_tensor * blocks = op->src[2];

    return (query->type == GGML_TYPE_F32 || query->type == GGML_TYPE_F16 || query->type == GGML_TYPE_BF16) &&
        (!blocks || blocks->ne[1] <= INKLING2_MIX_MAX_BLOCKS) && op->ne[2] <= 65535;
}

void ggml_cuda_op_inkling2_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x      = dst->src[0];
    const ggml_tensor * query  = dst->src[1];
    const ggml_tensor * blocks = dst->src[2];

    float scale;
    float eps;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&eps,   (const float *) dst->op_params + 1, sizeof(float));

    const int n        = x->ne[0];
    const int T        = x->ne[1];
    const int nq       = query->ne[1];
    const int n_blocks = blocks ? blocks->ne[1] : 0;

    // decode runs only nq blocks: a big block keeps more loads in flight
    const dim3 grid(nq, T);
    const dim3 threads(1024);

    const float * b_d  = blocks ? (const float *) blocks->data : nullptr;
    const int64_t s_b1 = blocks ? blocks->nb[1]/sizeof(float) : 0;
    const int64_t s_b2 = blocks ? blocks->nb[2]/sizeof(float) : 0;
    const int64_t s_x1 = x->nb[1]/sizeof(float);
    const int64_t s_q1 = query->nb[1]/ggml_type_size(query->type);

    switch (query->type) {
        case GGML_TYPE_F32:
            inkling2_mix_f32<<<grid, threads, 0, ctx.stream()>>>((const float *) x->data, (const float *) query->data, b_d, (float *) dst->data,
                    n, nq, n_blocks, s_x1, s_q1, s_b1, s_b2, scale, eps);
            break;
        case GGML_TYPE_F16:
            inkling2_mix_f32<<<grid, threads, 0, ctx.stream()>>>((const float *) x->data, (const half *) query->data, b_d, (float *) dst->data,
                    n, nq, n_blocks, s_x1, s_q1, s_b1, s_b2, scale, eps);
            break;
        case GGML_TYPE_BF16:
            inkling2_mix_f32<<<grid, threads, 0, ctx.stream()>>>((const float *) x->data, (const nv_bfloat16 *) query->data, b_d, (float *) dst->data,
                    n, nq, n_blocks, s_x1, s_q1, s_b1, s_b2, scale, eps);
            break;
        default:
            GGML_ABORT("unsupported type %s", ggml_type_name(query->type));
    }
}

// one thread per kv cell, one block row per (head, token)
template <typename T, typename M>
static __global__ void inkling2_rel_bias_f32(
        const T * __restrict__ table, const float * __restrict__ r, const float * __restrict__ sinks, const int32_t * __restrict__ rel_pos,
        const M * __restrict__ mask, float * __restrict__ dst, const int n_rel, const int n_extent, const int n_kv, const int n_ps,
        const int64_t s_table1, const int64_t s_r1, const int64_t s_r2, const int64_t s_sinks1, const int64_t s_pos1,
        const int64_t s_mask1, const int64_t s_mask3, const int64_t s_dst1, const int64_t s_dst2, const int64_t s_dst3) {
    extern __shared__ float r_s[];

    const int h = blockIdx.y;
    const int t = blockIdx.z;

    for (int k = threadIdx.x; k < n_rel; k += blockDim.x) {
        r_s[k] = r[h*s_r1 + t*s_r2 + k];
    }
    __syncthreads();

    const int kv = blockIdx.x*blockDim.x + threadIdx.x;
    if (kv >= n_kv) {
        return;
    }

    const int d = rel_pos[t*s_pos1 + kv];

    float v = 0.0f;
    if (d < n_extent) {
        const T * col = table + d*s_table1;
        for (int k = 0; k < n_rel; ++k) {
            v += ggml_cuda_cast<float>(col[k])*r_s[k];
        }
    }
    v -= sinks[t*s_sinks1 + h];
    if constexpr (!std::is_same_v<M, void>) {
        v += ggml_cuda_cast<float>(mask[(t % n_ps)*s_mask1 + (t / n_ps)*s_mask3 + kv]);
    }

    dst[(t % n_ps)*s_dst1 + h*s_dst2 + (t / n_ps)*s_dst3 + kv] = v;
}

bool ggml_cuda_inkling2_rel_bias_supported(const ggml_tensor * op) {
    const ggml_type type = op->src[0]->type;
    return (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) && op->src[1]->ne[1] <= 65535 && op->src[1]->ne[2] <= 65535;
}

void ggml_cuda_op_inkling2_rel_bias(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * table   = dst->src[0];
    const ggml_tensor * r       = dst->src[1];
    const ggml_tensor * sinks   = dst->src[2];
    const ggml_tensor * rel_pos = dst->src[3];
    const ggml_tensor * mask    = dst->src[4];

    const int n_rel    = table->ne[0];
    const int n_extent = table->ne[1];
    const int n_kv     = dst->ne[0];
    const int n_ps     = dst->ne[1];
    const int n_head   = dst->ne[2];
    const int T        = r->ne[2];

    const int  n_threads = 256;
    const dim3 grid((n_kv + n_threads - 1)/n_threads, n_head, T);
    const size_t smem = n_rel*sizeof(float);

    const int64_t s_table1 = table->nb[1]/ggml_type_size(table->type);

    const int64_t s_mask1 = mask ? mask->nb[1]/ggml_type_size(mask->type) : 0;
    const int64_t s_mask3 = mask ? mask->nb[3]/ggml_type_size(mask->type) : 0;

    auto launch = [&](auto kernel, const auto * table_d, const auto * mask_d) {
        kernel<<<grid, n_threads, smem, ctx.stream()>>>(table_d, (const float *) r->data, (const float *) sinks->data,
                (const int32_t *) rel_pos->data, mask_d, (float *) dst->data, n_rel, n_extent, n_kv, n_ps, s_table1,
                r->nb[1]/sizeof(float), r->nb[2]/sizeof(float), sinks->nb[1]/sizeof(float), rel_pos->nb[1]/sizeof(int32_t),
                s_mask1, s_mask3, dst->nb[1]/sizeof(float), dst->nb[2]/sizeof(float), dst->nb[3]/sizeof(float));
    };
    auto launch_mask = [&](auto t_tag, const auto * table_d) {
        using T = decltype(t_tag);
        if (!mask) {
            launch(inkling2_rel_bias_f32<T, void>, table_d, (const void *) nullptr);
        } else if (mask->type == GGML_TYPE_F16) {
            launch(inkling2_rel_bias_f32<T, half>, table_d, (const half *) mask->data);
        } else {
            launch(inkling2_rel_bias_f32<T, float>, table_d, (const float *) mask->data);
        }
    };

    switch (table->type) {
        case GGML_TYPE_F32:  launch_mask(float(),       (const float *)       table->data); break;
        case GGML_TYPE_F16:  launch_mask(half(),        (const half *)        table->data); break;
        case GGML_TYPE_BF16: launch_mask(nv_bfloat16(), (const nv_bfloat16 *) table->data); break;
        default: GGML_ABORT("unsupported type %s", ggml_type_name(table->type));
    }
}

// one thread per element; gate and up may be strided views
static __global__ void inkling2_situ_f32(
        const float * __restrict__ gate, const float * __restrict__ up, float * __restrict__ dst, const int64_t ne,
        const int ne0, const int ne1, const int ne2, const int64_t sg1, const int64_t sg2, const int64_t sg3,
        const int64_t su1, const int64_t su2, const int64_t su3, const float beta, const float lb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= ne) {
        return;
    }
    const int64_t i0 = i % ne0;
    const int64_t i1 = (i / ne0) % ne1;
    const int64_t i2 = (i / ((int64_t) ne0*ne1)) % ne2;
    const int64_t i3 = i / ((int64_t) ne0*ne1*ne2);

    const float g = gate[i0 + i1*sg1 + i2*sg2 + i3*sg3];
    const float u = up  [i0 + i1*su1 + i2*su2 + i3*su3];

    const float a = tanhf(g*(1.0f/beta))*beta*(1.0f/(1.0f + expf(-g)));
    const float b = lb > 0.0f ? tanhf(u*(1.0f/lb))*lb : u;
    dst[i] = a*b;
}

void ggml_cuda_op_inkling2_situ(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * gate = dst->src[0];
    const ggml_tensor * up   = dst->src[1];

    float beta;
    float lb;
    memcpy(&beta, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&lb,   (const float *) dst->op_params + 1, sizeof(float));

    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t ne = ggml_nelements(dst);
    const int     n_threads = 256;

    inkling2_situ_f32<<<(ne + n_threads - 1)/n_threads, n_threads, 0, ctx.stream()>>>((const float *) gate->data, (const float *) up->data,
            (float *) dst->data, ne, gate->ne[0], gate->ne[1], gate->ne[2],
            gate->nb[1]/sizeof(float), gate->nb[2]/sizeof(float), gate->nb[3]/sizeof(float),
            up->nb[1]/sizeof(float), up->nb[2]/sizeof(float), up->nb[3]/sizeof(float), beta, lb);
}
