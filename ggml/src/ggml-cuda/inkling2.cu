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

// one input of the clip kernel
struct inkling2_clip_src {
    const float * x;
    const float * scale;
    float       * dst;
    int           ne00, ne01, ne02;
    int64_t       nr, s01, s02, s03, s_scale1;
    float         inv_threshold, eps_threshold;
};

// dst = y/max(1, (|y| + eps)/threshold)*tau with y = x*scale, one warp per row; rows of a, then rows of b
template <bool has_scale, bool has_tau>
static __global__ void inkling2_clip_f32(const inkling2_clip_src a, const inkling2_clip_src b, const float * __restrict__ tau, const int s_tau2) {
    int64_t ir = (int64_t) blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    if (ir >= a.nr + b.nr) {
        return;
    }
    const inkling2_clip_src & p = ir < a.nr ? a : b;
    if (ir >= a.nr) {
        ir -= a.nr;
    }
    const int lane = threadIdx.x % WARP_SIZE;

    const int64_t i1 = ir % p.ne01;
    const int64_t i2 = (ir/p.ne01) % p.ne02;
    const int64_t i3 = ir/((int64_t) p.ne01*p.ne02);

    const float * xs = p.x + i1*p.s01 + i2*p.s02 + i3*p.s03;
    const float * ss = has_scale ? p.scale + i1*p.s_scale1 : nullptr;
    float       * y  = p.dst + ir*p.ne00;

    float sum = 0.0f;
    for (int i0 = lane; i0 < p.ne00; i0 += WARP_SIZE) {
        const float v = has_scale ? xs[i0]*ss[i0] : xs[i0];
        sum += v*v;
    }
    sum = warp_reduce_sum(sum);

    const float d = fmaxf(sqrtf(sum)*p.inv_threshold + p.eps_threshold, 1.0f);
    const float t = has_tau ? tau[i2*s_tau2] : 1.0f;

    for (int i0 = lane; i0 < p.ne00; i0 += WARP_SIZE) {
        const float v = has_scale ? xs[i0]*ss[i0] : xs[i0];
        y[i0] = v/d*t;
    }
}

static inkling2_clip_src inkling2_clip_make_src(const ggml_tensor * x, const ggml_tensor * scale, float * dst, float threshold, float eps) {
    inkling2_clip_src p;
    p.x             = (const float *) x->data;
    p.scale         = scale ? (const float *) scale->data : nullptr;
    p.dst           = dst;
    p.ne00          = x->ne[0];
    p.ne01          = x->ne[1];
    p.ne02          = x->ne[2];
    p.nr            = ggml_nrows(x);
    p.s01           = x->nb[1]/sizeof(float);
    p.s02           = x->nb[2]/sizeof(float);
    p.s03           = x->nb[3]/sizeof(float);
    p.s_scale1      = scale && scale->ne[1] > 1 ? x->ne[0] : 0;
    p.inv_threshold = 1.0f/threshold;
    p.eps_threshold = eps/threshold;
    return p;
}

void ggml_cuda_op_inkling2_clip(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x      = dst->src[0];
    const ggml_tensor * scale  = dst->src[1];
    const ggml_tensor * tau    = dst->src[2];
    const ggml_tensor * x2     = dst->src[3];
    const ggml_tensor * scale2 = dst->src[4];

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(!x2 || !scale == !scale2);

    const float threshold  = ggml_get_op_params_f32(dst, 0);
    const float eps        = ggml_get_op_params_f32(dst, 1);
    const float threshold2 = ggml_get_op_params_f32(dst, 2);

    float * y = (float *) dst->data;
    const inkling2_clip_src a = inkling2_clip_make_src(x, scale, y, threshold, eps);
    inkling2_clip_src       b = {};
    if (x2) {
        b = inkling2_clip_make_src(x2, scale2, y + ggml_nelements(x), threshold2, eps);
    }

    const int64_t nr             = a.nr + b.nr;
    const int     rows_per_block = 256/WARP_SIZE;
    const dim3    blocks((nr + rows_per_block - 1)/rows_per_block);
    const dim3    threads(rows_per_block*WARP_SIZE);
    const int     s_tau2 = tau && tau->ne[2] > 1 ? 1 : 0;

    auto kernel = scale ? (tau ? inkling2_clip_f32<true,  true> : inkling2_clip_f32<true,  false>)
                        : (tau ? inkling2_clip_f32<false, true> : inkling2_clip_f32<false, false>);
    kernel<<<blocks, threads, 0, ctx.stream()>>>(a, b, tau ? (const float *) tau->data : nullptr, s_tau2);
}

// tokens of one sequence per block in the conv kernel
static constexpr int INKLING2_CONV_CHUNK = 64;

// one thread per channel c of sequence blockIdx.y, tokens of chunk blockIdx.z; the last chunk also writes the new state
template <bool has_dynamic, bool has_residual>
static __global__ void inkling2_conv_f32(
        const float * __restrict__ cur, const float * __restrict__ past, const float * __restrict__ weight, const float * __restrict__ dynamic,
        const float * __restrict__ residual, const int32_t * __restrict__ ids, float * __restrict__ y, float * state,
        const int C, const int K, const int n_st, const int n_cph, const int offset,
        const int64_t s_cur1, const int64_t s_past0, const int64_t s_past1, const int64_t s_past2, const int64_t s_dyn1, const int64_t s_res1,
        const int64_t s_state_q, const float scale, const int post, const int hd, const float post_param) {
    // every thread of the block takes part in the head reductions of the post op, so none returns early
    const int  c_raw = blockIdx.x*blockDim.x + threadIdx.x;
    const bool valid = c_raw < C;
    const int  c     = valid ? c_raw : C - 1;
    const int q  = blockIdx.y;
    const int i0 = blockIdx.z*INKLING2_CONV_CHUNK;
    const int i1 = min(i0 + INKLING2_CONV_CHUNK, n_st);

    // with ids the past is read from cell ids[q] of the states, at channel offset `offset`
    const float * xp = ids ? past + ids[q]*s_past1 + (int64_t) (offset + c)*(K - 1) : past + c*s_past1 + q*s_past2;
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
        float       yv  = has_residual ? residual[t*s_res1 + c] + out : out;

        if (post != GGML_INKLING2_HEAD_POST_NONE) {
            // sum of squares of the head of channel c: hd/WARP_SIZE warps, all in this block
            __shared__ float hs[1024/WARP_SIZE];
            const float ss = warp_reduce_sum(yv*yv);
            if (threadIdx.x % WARP_SIZE == 0) {
                hs[threadIdx.x / WARP_SIZE] = ss;
            }
            __syncthreads();
            float sum_h = 0.0f;
            const int w0 = (threadIdx.x / hd)*(hd / WARP_SIZE);
            for (int w = 0; w < hd / WARP_SIZE; ++w) {
                sum_h += hs[w0 + w];
            }
            __syncthreads();
            if (post == GGML_INKLING2_HEAD_POST_CLIP) {
                yv = yv/fmaxf(sqrtf(sum_h)*(1.0f/post_param) + 1e-6f/post_param, 1.0f);
            } else {
                yv = yv*rsqrtf(sum_h/hd + post_param);
            }
        }

        if (valid) {
            y[t*C + c] = yv;
        }
    }

    if (valid && i1 == n_st) {
        for (int k = 0; k < K - 1; ++k) {
            state[q*s_state_q + (int64_t) c*(K - 1) + k] = x(n_st + k);
        }
    }
}

void ggml_cuda_op_inkling2_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_dst, int64_t s_state_q) {
    const ggml_tensor * cur     = dst->src[0];
    const ggml_tensor * past    = dst->src[1];
    const ggml_tensor * weight  = dst->src[2];
    const ggml_tensor * dynamic  = dst->src[3];
    const ggml_tensor * residual = dst->src[4];
    const ggml_tensor * ids      = dst->src[5];

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));
    const int offset = ggml_get_op_params_i32(dst, 1);

    const int C      = cur->ne[0];
    const int T      = cur->ne[1];
    const int K      = weight->ne[0];
    const int n_seqs = ids ? ids->ne[0] : past->ne[2];
    const int n_st   = T/n_seqs;
    const int n_cph  = dynamic ? C/(dynamic->ne[0]/K) : 1;

    float * y = (float *) dst->data;

    const int  n_threads = 256;
    const dim3 blocks((C + n_threads - 1)/n_threads, n_seqs, (n_st + INKLING2_CONV_CHUNK - 1)/INKLING2_CONV_CHUNK);

    const int   post       = ggml_get_op_params_i32(dst, 2);
    const int   hd         = ggml_get_op_params_i32(dst, 3);
    const float post_param = ggml_get_op_params_f32(dst, 4);
    GGML_ASSERT(post == GGML_INKLING2_HEAD_POST_NONE || (n_threads % hd == 0 && hd % WARP_SIZE == 0));

    auto kernel = dynamic ? (residual ? inkling2_conv_f32<true,  true> : inkling2_conv_f32<true,  false>)
                          : (residual ? inkling2_conv_f32<false, true> : inkling2_conv_f32<false, false>);
    float * state = state_dst ? state_dst : y + (int64_t) C*T;
    if (!state_dst) {
        s_state_q = (int64_t) C*(K - 1);
    }

    kernel<<<blocks, n_threads, 0, ctx.stream()>>>((const float *) cur->data, (const float *) past->data, (const float *) weight->data,
            dynamic ? (const float *) dynamic->data : nullptr, residual ? (const float *) residual->data : nullptr,
            ids ? (const int32_t *) ids->data : nullptr, y, state, C, K, n_st, n_cph, offset,
            cur->nb[1]/sizeof(float), past->nb[0]/sizeof(float), past->nb[1]/sizeof(float), ids ? 0 : past->nb[2]/sizeof(float),
            dynamic ? dynamic->nb[1]/sizeof(float) : 0, residual ? residual->nb[1]/sizeof(float) : 0, s_state_q, scale, post, hd, post_param);
}

// a conv followed by the copy of its new state into the cache can write the state there directly, when one block reads
// the past and writes the state of a sequence (one token chunk), or the past is not read from the cache
int ggml_cuda_inkling2_conv_state_fusion(const ggml_cgraph * cgraph, int node_idx, float ** state_dst, int64_t * s_state_q) {
    const ggml_tensor * conv = cgraph->nodes[node_idx];
    if (conv->op != GGML_OP_INKLING2_CONV || (conv->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * ids    = conv->src[5];
    const int64_t       C      = conv->src[0]->ne[0];
    const int64_t       T      = conv->src[0]->ne[1];
    const int64_t       K      = conv->src[2]->ne[0];
    const int64_t       n_seqs = ids ? ids->ne[0] : conv->src[1]->ne[2];

    if (ids && T/n_seqs > INKLING2_CONV_CHUNK) {
        return 0;
    }

    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_is_empty(n) || n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_NONE) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    // src: the state tail of this conv; dst: [(K - 1)*C, n_seqs] cache rows
    const ggml_tensor * src = cpy->src[0];
    const ggml_tensor * dst = cpy->src[1];
    if (src->view_src != conv || src->view_offs != (size_t) (C*T*sizeof(float)) || !ggml_is_contiguous(src) ||
        ggml_nelements(src) != (K - 1)*C*n_seqs) {
        return 0;
    }
    if (dst->type != GGML_TYPE_F32 || dst->data == nullptr || dst->ne[0] != (K - 1)*C || dst->ne[1] != n_seqs ||
        dst->ne[2] != 1 || dst->ne[3] != 1 || dst->nb[0] != sizeof(float) || dst->nb[1] % sizeof(float) != 0) {
        return 0;
    }

    *state_dst = (float *) dst->data;
    *s_state_q = dst->nb[1]/sizeof(float);
    return skip;
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

// single-pass vectorized mix for small n_blocks and n/4 <= 1024: each thread holds one float4 of x and each block in registers
template <typename T, int NB>
static __global__ void inkling2_mix_f32_vec(
        const float * __restrict__ x, const T * __restrict__ query, const float * __restrict__ blocks, float * __restrict__ dst,
        const int n, const int nq, const int64_t s_x1, const int64_t s_q1, const int64_t s_b1, const int64_t s_b2,
        const float scale, const float eps) {
    const int iq = blockIdx.x;
    const int t  = blockIdx.y;
    const int n4 = n / 4;
    const int i4 = threadIdx.x;

    const float * xs = x + t*s_x1;
    const T     * qs = query + iq*s_q1;
    const float * bs = blocks + t*s_b2;

    float4 xv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 bv[NB > 0 ? NB : 1];
    float  dot[NB > 0 ? NB : 1];
#pragma unroll
    for (int j = 0; j < NB; ++j) {
        dot[j] = 0.0f;
    }
    float ss = 0.0f;
    float dx = 0.0f;

    if (i4 < n4) {
        xv = ((const float4 *) xs)[i4];
        float4 qv;
        if constexpr (std::is_same_v<T, float>) {
            qv = ((const float4 *) qs)[i4];
        } else {
            const int i = 4*i4;
            qv = make_float4(ggml_cuda_cast<float>(qs[i + 0]), ggml_cuda_cast<float>(qs[i + 1]),
                             ggml_cuda_cast<float>(qs[i + 2]), ggml_cuda_cast<float>(qs[i + 3]));
        }
        ss = xv.x*xv.x + xv.y*xv.y + xv.z*xv.z + xv.w*xv.w;
        dx = xv.x*qv.x + xv.y*qv.y + xv.z*qv.z + xv.w*qv.w;
#pragma unroll
        for (int j = 0; j < NB; ++j) {
            const float4 b = ((const float4 *) (bs + j*s_b1))[i4];
            bv[j]  = b;
            dot[j] = b.x*qv.x + b.y*qv.y + b.z*qv.z + b.w*qv.w;
        }
    }

    __shared__ float red[NB + 2][32];
    __shared__ float p[NB + 2];

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int nw   = blockDim.x / WARP_SIZE;

#pragma unroll
    for (int j = 0; j < NB; ++j) {
        const float v = warp_reduce_sum(dot[j]);
        if (lane == 0) {
            red[j][warp] = v;
        }
    }
    ss = warp_reduce_sum(ss);
    dx = warp_reduce_sum(dx);
    if (lane == 0) {
        red[NB + 0][warp] = ss;
        red[NB + 1][warp] = dx;
    }
    __syncthreads();

    if (warp == 0) {
        float sum_ss = 0.0f;
        float sum_dx = 0.0f;
        float d      = 0.0f;
        for (int w = 0; w < nw; ++w) {
            sum_ss += red[NB + 0][w];
            sum_dx += red[NB + 1][w];
            if (lane < NB) {
                d += red[lane][w];
            }
        }
        const float s = rsqrtf(sum_ss/n + eps);
        if (lane == NB) {
            d = sum_dx*s;
        }

        const float sc  = lane <= NB ? d*scale : -INFINITY;
        const float max = warp_reduce_max(sc);
        const float e   = lane <= NB ? expf(sc - max) : 0.0f;
        const float sum = warp_reduce_sum(e);
        if (lane <= NB) {
            p[lane] = e/sum;
        }
        if (lane == 0) {
            p[NB + 1] = s;
        }
    }
    __syncthreads();

    if (i4 < n4) {
        const float sx = p[NB]*p[NB + 1];
        float4 acc = make_float4(xv.x*sx, xv.y*sx, xv.z*sx, xv.w*sx);
#pragma unroll
        for (int j = 0; j < NB; ++j) {
            const float pj = p[j];
            acc.x += bv[j].x*pj;
            acc.y += bv[j].y*pj;
            acc.z += bv[j].z*pj;
            acc.w += bv[j].w*pj;
        }
        ((float4 *) (dst + ((int64_t) t*nq + iq)*n))[i4] = acc;
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

    const dim3 grid(nq, T);

    const float * b_d  = blocks ? (const float *) blocks->data : nullptr;
    const int64_t s_b1 = blocks ? blocks->nb[1]/sizeof(float) : 0;
    const int64_t s_b2 = blocks ? blocks->nb[2]/sizeof(float) : 0;
    const int64_t s_x1 = x->nb[1]/sizeof(float);
    const int64_t s_q1 = query->nb[1]/ggml_type_size(query->type);

    const bool fast_vec = n % 4 == 0 && n/4 <= 1024 && n_blocks <= 6 &&
        inkling2_aligned(x->data, x->nb[1]) && inkling2_aligned(dst->data, (size_t) n*sizeof(float)) &&
        (query->type != GGML_TYPE_F32 || inkling2_aligned(query->data, query->nb[1])) &&
        (!blocks || (inkling2_aligned(blocks->data, blocks->nb[1]) && blocks->nb[2] % 16 == 0));

    if (fast_vec) {
        const int n_threads = ((n/4 + 63)/64)*64;
        auto launch_nb = [&](auto tag) {
            using T = decltype(tag);
            const T * q_d = (const T *) query->data;
            auto go = [&](auto nb_tag) {
                constexpr int NB = decltype(nb_tag)::value;
                inkling2_mix_f32_vec<T, NB><<<grid, n_threads, 0, ctx.stream()>>>((const float *) x->data, q_d, b_d, (float *) dst->data,
                        n, nq, s_x1, s_q1, s_b1, s_b2, scale, eps);
            };
            switch (n_blocks) {
                case 0: go(std::integral_constant<int, 0>()); break;
                case 1: go(std::integral_constant<int, 1>()); break;
                case 2: go(std::integral_constant<int, 2>()); break;
                case 3: go(std::integral_constant<int, 3>()); break;
                case 4: go(std::integral_constant<int, 4>()); break;
                case 5: go(std::integral_constant<int, 5>()); break;
                case 6: go(std::integral_constant<int, 6>()); break;
                default: GGML_ABORT("unexpected n_blocks %d", n_blocks);
            }
        };
        switch (query->type) {
            case GGML_TYPE_F32:  launch_nb(float());       return;
            case GGML_TYPE_F16:  launch_nb(half());        return;
            case GGML_TYPE_BF16: launch_nb(nv_bfloat16()); return;
            default: GGML_ABORT("unsupported type %s", ggml_type_name(query->type));
        }
    }

    // decode runs only nq blocks: a big block keeps more loads in flight
    const dim3 threads(1024);

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

static constexpr int INKLING2_PROJ_MAX_W          = GGML_MAX_SRC - 2;
static constexpr int INKLING2_PROJ_ROWS_PER_BLOCK = 32;

static __host__ __device__ __forceinline__ int inkling2_proj_xs_idx(int i) {
    return i + ((i >> 5) << 2);
}

struct inkling2_proj_weights {
    const void * w[INKLING2_PROJ_MAX_W];
    int64_t      s_w[INKLING2_PROJ_MAX_W];       // row stride in elements
    int          rows[INKLING2_PROJ_MAX_W];
    int          out_off[INKLING2_PROJ_MAX_W];   // first output row
    int          blk_start[INKLING2_PROJ_MAX_W + 1];
    int          rpb;                            // rows per block
};

// the input row (normalized) in LDS, then one warp per nr weight rows in parallel
template <typename T, bool vec, bool has_norm>
static __global__ void inkling2_proj_f32(
        const float * __restrict__ x, const float * __restrict__ norm, float * __restrict__ dst, const inkling2_proj_weights p,
        const int n, const int64_t s_x1, const int64_t s_d1, const float eps) {
    constexpr int v  = vec ? 16/sizeof(T) : 1;
    constexpr int nw = 512 / WARP_SIZE;
    constexpr int nr = INKLING2_PROJ_ROWS_PER_BLOCK / nw;

    extern __shared__ float xs[];

    const int t = blockIdx.y;
    const float * xr = x + t*s_x1;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xi = xr[i];
        xs[inkling2_proj_xs_idx(i)] = xi;
        ss += xi*xi;
    }
    if constexpr (has_norm) {
        __shared__ float buf[32];
        ss = block_reduce<block_reduce_method::SUM>(ss, buf);
        const float scale = rsqrtf(ss/n + eps);
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const int idx = inkling2_proj_xs_idx(i);
            xs[idx] = (xs[idx]*scale)*norm[i];
        }
    }
    __syncthreads();

    int k = 0;
    while ((int) blockIdx.x >= p.blk_start[k + 1]) {
        ++k;
    }
    const T *     w    = (const T *) p.w[k];
    const int64_t sw   = p.s_w[k];
    const int     r0   = (blockIdx.x - p.blk_start[k])*p.rpb;
    const int     r1   = min(r0 + p.rpb, p.rows[k]);
    const int     warp = threadIdx.x / WARP_SIZE;
    const int     lane = threadIdx.x % WARP_SIZE;

    if constexpr (vec) {
        if (r1 - r0 == nr*nw) {
            float acc[nr];
#pragma unroll
            for (int j = 0; j < nr; ++j) {
                acc[j] = 0.0f;
            }
            for (int c = lane*v; c < n; c += WARP_SIZE*v) {
                // issue all nr 16-byte weight loads before unpacking and FMA
                int4 raw[nr];
#pragma unroll
                for (int j = 0; j < nr; ++j) {
                    raw[j] = *(const int4 *) (w + (r0 + warp + j*nw)*sw + c);
                }
                const float * xsp = xs + inkling2_proj_xs_idx(c);
                float xv[v];
#pragma unroll
                for (int q = 0; q < v/4; ++q) {
                    ((float4 *) xv)[q] = ((const float4 *) xsp)[q];
                }
#pragma unroll
                for (int j = 0; j < nr; ++j) {
                    const T * wt = (const T *) &raw[j];
#pragma unroll
                    for (int q = 0; q < v; ++q) {
                        acc[j] += ggml_cuda_cast<float>(wt[q])*xv[q];
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < nr; ++j) {
                const float val = warp_reduce_sum(acc[j]);
                if (lane == 0) {
                    dst[t*s_d1 + p.out_off[k] + r0 + warp + j*nw] = val;
                }
            }
            return;
        }
    }

    for (int r = r0 + warp; r < r1; r += nw) {
        const T * wr = w + r*sw;
        float acc = 0.0f;
#pragma unroll 4
        for (int c = lane*v; c < n; c += WARP_SIZE*v) {
            float wf[v];
            inkling2_load<v>(wr + c, wf);
#pragma unroll
            for (int q = 0; q < v; ++q) {
                acc += wf[q]*xs[inkling2_proj_xs_idx(c + q)];
            }
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0) {
            dst[t*s_d1 + p.out_off[k] + r] = acc;
        }
    }
}

bool ggml_cuda_inkling2_proj_supported(const ggml_tensor * op) {
    const ggml_type type = op->src[1]->type;
    const int64_t   n    = op->src[0]->ne[0];
    return (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) && op->ne[1] <= 65535 &&
        (size_t) inkling2_proj_xs_idx((int) n)*sizeof(float) <= 48*1024;
}

void ggml_cuda_op_inkling2_proj(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];

    float eps;
    memcpy(&eps, (const float *) dst->op_params + 0, sizeof(float));
    const int n_w = ggml_get_op_params_i32(dst, 1);

    const ggml_tensor * norm = dst->src[1 + n_w];
    const ggml_type     type = dst->src[1]->type;
    const size_t        ts   = ggml_type_size(type);
    const int           n    = x->ne[0];

    // each block normalizes the input again, so a block takes many rows
    inkling2_proj_weights p = {};
    p.rpb = INKLING2_PROJ_ROWS_PER_BLOCK;
    bool aligned = n % (16/ts) == 0;
    int  off     = 0;
    for (int i = 0; i < n_w; ++i) {
        const ggml_tensor * w = dst->src[1 + i];
        p.w[i]             = w->data;
        p.s_w[i]           = w->nb[1]/ts;
        p.rows[i]          = w->ne[1];
        p.out_off[i]       = off;
        p.blk_start[i + 1] = p.blk_start[i] + (w->ne[1] + p.rpb - 1)/p.rpb;
        off               += w->ne[1];
        aligned            = aligned && inkling2_aligned(w->data, w->nb[1]);
    }
    for (int i = n_w; i < INKLING2_PROJ_MAX_W; ++i) {
        p.blk_start[i + 1] = p.blk_start[n_w];
    }

    const dim3   grid(p.blk_start[n_w], x->ne[1]);
    const int    n_threads = 512;
    const size_t smem      = inkling2_proj_xs_idx(n)*sizeof(float);

    auto launch = [&](auto kernel) {
        kernel<<<grid, n_threads, smem, ctx.stream()>>>((const float *) x->data, norm ? (const float *) norm->data : nullptr,
                (float *) dst->data, p, n, x->nb[1]/sizeof(float), dst->nb[1]/sizeof(float), eps);
    };
    auto launch_t = [&](auto tag) {
        using T = decltype(tag);
        if (aligned) {
            norm ? launch(inkling2_proj_f32<T, true,  true>) : launch(inkling2_proj_f32<T, true,  false>);
        } else {
            norm ? launch(inkling2_proj_f32<T, false, true>) : launch(inkling2_proj_f32<T, false, false>);
        }
    };

    switch (type) {
        case GGML_TYPE_F32:  launch_t(float());       break;
        case GGML_TYPE_F16:  launch_t(half());        break;
        case GGML_TYPE_BF16: launch_t(nv_bfloat16()); break;
        default: GGML_ABORT("unsupported type %s", ggml_type_name(type));
    }
}

// the most kv cells of the fused attention: the logits stay in LDS
static constexpr int INKLING2_ATTN_MAX_KV = 8192;

// in-place normalized Walsh-Hadamard transform of block size n_rot over x[0..d-1] in LDS
static __device__ __forceinline__ void inkling2_fwht_smem(float * x, const int d, const int n_rot) {
    if (n_rot <= 0) {
        return;
    }
    constexpr int pws = ggml_cuda_get_physical_warp_size();
    const int i = threadIdx.x;
    if (n_rot == 2*pws) {
        const int warp = i / pws;
        const int lane = i % pws;
        const int off  = warp*n_rot;
        if (off < d) {
            const float inv = rsqrtf((float) n_rot);
            float a = x[off + lane]*inv;
            float b = x[off + lane + pws]*inv;
#pragma unroll
            for (int h = 1; h < pws; h *= 2) {
                const float a2 = ggml_cuda_shfl_xor(a, h, pws);
                const float b2 = ggml_cuda_shfl_xor(b, h, pws);
                a = (lane & h) == 0 ? a + a2 : a2 - a;
                b = (lane & h) == 0 ? b + b2 : b2 - b;
            }
            x[off + lane]       = a + b;
            x[off + lane + pws] = a - b;
        }
        __syncthreads();
        return;
    }
    for (int h = 1; h < n_rot; h *= 2) {
        float val = 0.0f;
        if (i < d) {
            const float a = x[i];
            const float b = x[i ^ h];
            val = (i & h) == 0 ? a + b : b - a;
        }
        __syncthreads();
        if (i < d) {
            x[i] = val;
        }
        __syncthreads();
    }
    if (i < d) {
        x[i] *= rsqrtf((float) n_rot);
    }
    __syncthreads();
}

// one block per query head: the logits (16 lanes per kv cell, a contiguous K row), the softmax with a zero sink,
// then the V sum (one warp per output dim, the lanes walk the keys of a transposed V row)
template <typename T, typename M, bool v_trans, bool vec, bool vec_v>
static __global__ void inkling2_attn_f32(
        const float * __restrict__ q, const T * __restrict__ k, const T * __restrict__ v, const float * __restrict__ table,
        const float * __restrict__ r, const float * __restrict__ sinks, const int32_t * __restrict__ rel_pos, const M * __restrict__ mask,
        const float * __restrict__ gate, float * __restrict__ dst, const int d, const int n_gqa, const int n_kv, const int n_rel, const int n_extent,
        const int64_t s_q1, const int64_t s_k1, const int64_t s_k2, const int64_t s_v1, const int64_t s_v2, const int64_t s_t1, const int64_t s_r1,
        const float scale, const int n_rot_q, const int n_rot_v) {
    constexpr int vk    = vec   ? 16/sizeof(T) : 1;
    constexpr int vv    = vec_v ? 16/sizeof(T) : 1;
    constexpr int lanes = 16;

    extern __shared__ float sm[];
    float * lg = sm;
    float * qs = lg + n_kv;
    float * rs = qs + d;

    const int h  = blockIdx.x;
    const int hk = h/n_gqa;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        qs[i] = q[h*s_q1 + i];
    }
    for (int i = threadIdx.x; i < n_rel; i += blockDim.x) {
        rs[i] = r[h*s_r1 + i];
    }
    __syncthreads();
    inkling2_fwht_smem(qs, d, n_rot_q);

    const float sh   = sinks[h];
    const int   li   = threadIdx.x % lanes;
    const int   ng   = blockDim.x / lanes;
    const int   warp = threadIdx.x / WARP_SIZE;
    const int   lane = threadIdx.x % WARP_SIZE;
    const int   nw   = blockDim.x / WARP_SIZE;

    // all lanes of a group take part in the shuffles, so the loop runs the same count in every group
    float mx = 0.0f; // the zero sink
    for (int j0 = 0; j0 < n_kv; j0 += ng) {
        const int  j     = j0 + threadIdx.x / lanes;
        const bool valid = j < n_kv;

        float part = 0.0f;
        int   dist = n_extent;
        if (valid) {
            const T * kj = k + hk*s_k1 + j*s_k2;
            float qk = 0.0f;
            for (int i = li*vk; i < d; i += lanes*vk) {
                float kf[vk];
                inkling2_load<vk>(kj + i, kf);
#pragma unroll
                for (int c = 0; c < vk; ++c) {
                    qk += qs[i + c]*kf[c];
                }
            }
            float b = 0.0f;
            dist = rel_pos[j];
            if (dist < n_extent) {
                for (int i = li; i < n_rel; i += lanes) {
                    b += table[dist*s_t1 + i]*rs[i];
                }
            }
            part = qk*scale + b;
        }
        part = warp_reduce_sum<lanes>(part);
        if (valid && li == 0) {
            const float l = (part - sh) + ggml_cuda_cast<float>(mask[j]);
            lg[j] = l;
            mx = fmaxf(mx, l);
        }
    }

    __shared__ float buf[32];
    mx = block_reduce<block_reduce_method::MAX>(mx, buf);
    __syncthreads();

    float sum = 0.0f;
    for (int j = threadIdx.x; j < n_kv; j += blockDim.x) {
        const float e = expf(lg[j] - mx);
        lg[j] = e;
        sum  += e;
    }
    sum  = block_reduce<block_reduce_method::SUM>(sum, buf);
    sum += expf(0.0f - mx);
    __syncthreads();

    if constexpr (v_trans) {
        for (int i = warp; i < d; i += nw) {
            float acc = 0.0f;
            const T * vi = v + hk*s_v1 + i*s_v2;
            int j = lane*vv;
            for (; j + vv <= n_kv; j += WARP_SIZE*vv) {
                float vf[vv];
                inkling2_load<vv>(vi + j, vf);
#pragma unroll
                for (int c = 0; c < vv; ++c) {
                    acc += lg[j + c]*vf[c];
                }
            }
            // the last, partial chunk of this lane
            for (const int jt = j; j < n_kv && j < jt + vv; ++j) {
                acc += lg[j]*ggml_cuda_cast<float>(vi[j]);
            }
            acc = warp_reduce_sum(acc);
            if (lane == 0) {
                if (n_rot_v > 0) {
                    qs[i] = acc/sum;
                } else {
                    const float gt = gate[h*d + i];
                    dst[h*d + i] = (acc/sum)*(1.0f/(1.0f + expf(-gt)));
                }
            }
        }
        if (n_rot_v > 0) {
            __syncthreads();
            inkling2_fwht_smem(qs, d, n_rot_v);
            if (threadIdx.x < d) {
                const float gt = gate[h*d + threadIdx.x];
                dst[h*d + threadIdx.x] = qs[threadIdx.x]*(1.0f/(1.0f + expf(-gt)));
            }
        }
    } else {
        // coalesced V sum: d consecutive threads cover one V row, ng_v groups walk the keys in parallel
        const int ng_v = blockDim.x / d;
        const int vg   = threadIdx.x / d;
        const int vi_i = threadIdx.x % d;
        float acc = 0.0f;
        if (vg < ng_v) {
            for (int j = vg; j < n_kv; j += ng_v) {
                acc += lg[j]*ggml_cuda_cast<float>(v[vi_i + hk*s_v1 + j*s_v2]);
            }
        }
        __shared__ float vbuf[512];
        if (vg < ng_v) {
            vbuf[vg*d + vi_i] = acc;
        }
        __syncthreads();
        if (threadIdx.x < d) {
            float total = 0.0f;
            for (int g = 0; g < ng_v; ++g) {
                total += vbuf[g*d + threadIdx.x];
            }
            qs[threadIdx.x] = total/sum;
        }
        __syncthreads();
        inkling2_fwht_smem(qs, d, n_rot_v);
        if (threadIdx.x < d) {
            const float gt = gate[h*d + threadIdx.x];
            dst[h*d + threadIdx.x] = qs[threadIdx.x]*(1.0f/(1.0f + expf(-gt)));
        }
    }
}

// Q8_0 KV cache (non-transposed V): 16 lanes per kv cell for K, coalesced d/4 threads per V row
template <typename M>
static __global__ void inkling2_attn_q8_0_f32(
        const float * __restrict__ q, const char * __restrict__ k, const char * __restrict__ v, const float * __restrict__ table,
        const float * __restrict__ r, const float * __restrict__ sinks, const int32_t * __restrict__ rel_pos, const M * __restrict__ mask,
        const float * __restrict__ gate, float * __restrict__ dst, const int d, const int n_gqa, const int n_kv, const int n_rel, const int n_extent,
        const int64_t s_q1, const int64_t s_k1, const int64_t s_k2, const int64_t s_v1, const int64_t s_v2, const int64_t s_t1, const int64_t s_r1,
        const float scale, const int n_rot_q, const int n_rot_v) {
    constexpr int lanes = 16;

    extern __shared__ float sm[];
    float * lg = sm;
    float * qs = lg + n_kv;
    float * rs = qs + d;

    const int h  = blockIdx.x;
    const int hk = h/n_gqa;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        qs[i] = q[h*s_q1 + i];
    }
    for (int i = threadIdx.x; i < n_rel; i += blockDim.x) {
        rs[i] = r[h*s_r1 + i];
    }
    __syncthreads();
    inkling2_fwht_smem(qs, d, n_rot_q);

    const float sh = sinks[h];
    const int   li = threadIdx.x % lanes;
    const int   ng = blockDim.x / lanes;
    const int   d4 = d / 4;

    // all lanes of a group take part in the shuffles, so the loop runs the same count in every group
    float mx = 0.0f; // the zero sink
    for (int j0 = 0; j0 < n_kv; j0 += ng) {
        const int  j     = j0 + threadIdx.x / lanes;
        const bool valid = j < n_kv;

        float part = 0.0f;
        if (valid) {
            const block_q8_0 * kj = (const block_q8_0 *) (k + hk*s_k1 + j*s_k2);
            float qk = 0.0f;
            for (int c = li; c < d4; c += lanes) {
                const int ib  = c / QI8_0;
                const int iqs = c % QI8_0;
                int q4;
                ggml_cuda_memcpy_1<sizeof(q4), 2>(&q4, kj[ib].qs + 4*iqs);
                const float kd  = __half2float(kj[ib].d);
                const float dot = qs[4*c + 0]*int8_t(q4 >>  0) + qs[4*c + 1]*int8_t(q4 >>  8) +
                                  qs[4*c + 2]*int8_t(q4 >> 16) + qs[4*c + 3]*int8_t(q4 >> 24);
                qk += dot*kd;
            }
            float b = 0.0f;
            const int dist = rel_pos[j];
            if (dist < n_extent) {
                for (int i = li; i < n_rel; i += lanes) {
                    b += table[dist*s_t1 + i]*rs[i];
                }
            }
            part = qk*scale + b;
        }
        part = warp_reduce_sum<lanes>(part);
        if (valid && li == 0) {
            const float l = (part - sh) + ggml_cuda_cast<float>(mask[j]);
            lg[j] = l;
            mx = fmaxf(mx, l);
        }
    }

    __shared__ float buf[32];
    mx = block_reduce<block_reduce_method::MAX>(mx, buf);
    __syncthreads();

    float sum = 0.0f;
    for (int j = threadIdx.x; j < n_kv; j += blockDim.x) {
        const float e = expf(lg[j] - mx);
        lg[j] = e;
        sum  += e;
    }
    sum  = block_reduce<block_reduce_method::SUM>(sum, buf);
    sum += expf(0.0f - mx);
    __syncthreads();

    const int ng_v = blockDim.x / d4;
    const int vg   = threadIdx.x / d4;
    const int c    = threadIdx.x % d4;
    const int ib   = c / QI8_0;
    const int iqs  = c % QI8_0;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (vg < ng_v) {
        for (int j = vg; j < n_kv; j += ng_v) {
            const block_q8_0 * vj = (const block_q8_0 *) (v + hk*s_v1 + j*s_v2) + ib;
            int q4;
            ggml_cuda_memcpy_1<sizeof(q4), 2>(&q4, vj->qs + 4*iqs);
            const float w = lg[j]*__half2float(vj->d);
            acc[0] += w*int8_t(q4 >>  0);
            acc[1] += w*int8_t(q4 >>  8);
            acc[2] += w*int8_t(q4 >> 16);
            acc[3] += w*int8_t(q4 >> 24);
        }
    }

    __shared__ float vbuf[2048];
    if (vg < ng_v) {
        float * vp = vbuf + vg*d + 4*c;
        vp[0] = acc[0];
        vp[1] = acc[1];
        vp[2] = acc[2];
        vp[3] = acc[3];
    }
    __syncthreads();
    if (threadIdx.x < d) {
        float total = 0.0f;
        for (int g = 0; g < ng_v; ++g) {
            total += vbuf[g*d + threadIdx.x];
        }
        qs[threadIdx.x] = total/sum;
    }
    __syncthreads();

    inkling2_fwht_smem(qs, d, n_rot_v);
    if (threadIdx.x < d) {
        const float gt = gate[h*d + threadIdx.x];
        dst[h*d + threadIdx.x] = qs[threadIdx.x]*(1.0f/(1.0f + expf(-gt)));
    }
}

bool ggml_cuda_inkling2_attn_supported(const ggml_tensor * op) {
    const ggml_type type    = op->src[1]->type;
    const int64_t   d       = op->src[0]->ne[0];
    const bool      v_trans = ggml_get_op_params_i32(op, 1) != 0;
    const bool      ok_type = type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 ||
        (type == GGML_TYPE_Q8_0 && !v_trans && d % QK8_0 == 0);
    return ok_type && op->src[3]->type == GGML_TYPE_F32 &&
        op->src[6]->ne[0] <= INKLING2_ATTN_MAX_KV && d <= 512;
}

void ggml_cuda_op_inkling2_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q       = dst->src[0];
    const ggml_tensor * k       = dst->src[1];
    const ggml_tensor * v       = dst->src[2];
    const ggml_tensor * table   = dst->src[3];
    const ggml_tensor * r       = dst->src[4];
    const ggml_tensor * sinks   = dst->src[5];
    const ggml_tensor * rel_pos = dst->src[6];
    const ggml_tensor * mask    = dst->src[7];
    const ggml_tensor * gate    = dst->src[8];

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    const bool v_trans = ggml_get_op_params_i32(dst, 1) != 0;
    const int  n_rot_q = ggml_get_op_params_i32(dst, 2);
    const int  n_rot_v = ggml_get_op_params_i32(dst, 3);

    const int d      = q->ne[0];
    const int n_head = q->ne[1];
    const int n_kv   = rel_pos->ne[0];
    const int n_rel  = table->ne[0];
    const int n_gqa  = n_head/k->ne[1];

    const int    n_threads = 512;
    const size_t smem      = (n_kv + d + n_rel)*sizeof(float);

    if (k->type == GGML_TYPE_Q8_0) {
        auto launch_q8 = [&](auto mask_tag) {
            using M = decltype(mask_tag);
            inkling2_attn_q8_0_f32<M><<<n_head, n_threads, smem, ctx.stream()>>>(
                (const float *) q->data, (const char *) k->data, (const char *) v->data, (const float *) table->data,
                (const float *) r->data, (const float *) sinks->data, (const int32_t *) rel_pos->data, (const M *) mask->data,
                (const float *) gate->data, (float *) dst->data,
                d, n_gqa, n_kv, n_rel, (int) table->ne[1], q->nb[1]/sizeof(float), k->nb[1], k->nb[2], v->nb[1], v->nb[2],
                table->nb[1]/sizeof(float), r->nb[1]/sizeof(float), scale, n_rot_q, n_rot_v);
        };
        mask->type == GGML_TYPE_F16 ? launch_q8(half()) : launch_q8(float());
        return;
    }

    const size_t ts = ggml_type_size(k->type);

    // 16-byte loads need aligned K rows, and aligned V rows when V is transposed
    const bool vec   = d % (16/ts) == 0 && inkling2_aligned(k->data, k->nb[2]) && k->nb[1] % 16 == 0;
    const bool vec_v = v_trans && inkling2_aligned(v->data, v->nb[2]) && v->nb[1] % 16 == 0;

    auto launch = [&](auto kernel, const auto * k_d, const auto * v_d, const auto * mask_d) {
        kernel<<<n_head, n_threads, smem, ctx.stream()>>>((const float *) q->data, k_d, v_d, (const float *) table->data, (const float *) r->data,
                (const float *) sinks->data, (const int32_t *) rel_pos->data, mask_d, (const float *) gate->data, (float *) dst->data,
                d, n_gqa, n_kv, n_rel, (int) table->ne[1], q->nb[1]/sizeof(float), k->nb[1]/ts, k->nb[2]/ts, v->nb[1]/ts, v->nb[2]/ts,
                table->nb[1]/sizeof(float), r->nb[1]/sizeof(float), scale, n_rot_q, n_rot_v);
    };
    auto launch_t = [&](auto tag) {
        using T = decltype(tag);
        const T * k_d = (const T *) k->data;
        const T * v_d = (const T *) v->data;
        auto launch_m = [&](auto mask_tag) {
            using M = decltype(mask_tag);
            const M * mask_d = (const M *) mask->data;
            if (!v_trans) {
                vec ? launch(inkling2_attn_f32<T, M, false, true, false>, k_d, v_d, mask_d) : launch(inkling2_attn_f32<T, M, false, false, false>, k_d, v_d, mask_d);
            } else if (vec_v) {
                vec ? launch(inkling2_attn_f32<T, M, true,  true, true>,  k_d, v_d, mask_d) : launch(inkling2_attn_f32<T, M, true,  false, true>,  k_d, v_d, mask_d);
            } else {
                vec ? launch(inkling2_attn_f32<T, M, true,  true, false>, k_d, v_d, mask_d) : launch(inkling2_attn_f32<T, M, true,  false, false>, k_d, v_d, mask_d);
            }
        };
        mask->type == GGML_TYPE_F16 ? launch_m(half()) : launch_m(float());
    };

    switch (k->type) {
        case GGML_TYPE_F32:  launch_t(float());       break;
        case GGML_TYPE_F16:  launch_t(half());        break;
        case GGML_TYPE_BF16: launch_t(nv_bfloat16()); break;
        default: GGML_ABORT("unsupported type %s", ggml_type_name(k->type));
    }
}
