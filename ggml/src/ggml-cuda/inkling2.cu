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

// one loop control op; up to INKLING2_CTRL_MAX of them run in one launch (blockIdx.z)
static constexpr int INKLING2_CTRL_MAX = 4;

struct inkling2_ctrl_args {
    const float * cur;
    const float * msg;
    const void  * down;
    const void  * gain;
    const void  * forget;
    float       * g;
    float       * dst;
    int64_t       s_cur, s_msg, s_down, s_gain, s_forget, s_dst;
};

struct inkling2_ctrl_batch {
    inkling2_ctrl_args a[INKLING2_CTRL_MAX];
};

static bool inkling2_aligned(const void * p, size_t stride) {
    return (uintptr_t) p % 16 == 0 && stride % 16 == 0;
}

// g[i, t] = down[i, :n].cur[t]/rms(cur[t]) + down[i, n:].msg[t]/rms(msg[t]), one block per (i, t, op)
template <typename T, bool vec>
static __global__ void inkling2_ctrl_down(const inkling2_ctrl_batch bt, const int n, const int r, const float eps) {
    constexpr int v = vec ? 16/sizeof(T) : 1;

    const inkling2_ctrl_args & p = bt.a[blockIdx.z];

    const int i = blockIdx.x;
    const int t = blockIdx.y;

    const float * c = p.cur + t*p.s_cur;
    const float * m = p.msg + t*p.s_msg;
    const T     * w = (const T *) p.down + i*p.s_down;

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
        p.g[t*r + i] = dot.x*rsqrtf(ss.x/n + eps) + dot.y*rsqrtf(ss.y/n + eps);
    }
}

// dst[j, t] = cur - cur*tanh(forget[j].g[t]) + msg*tanh(gain[j].g[t]), INKLING2_CTRL_UP_LANES lanes per (j, t, op)
template <typename T, bool vec>
static __global__ void inkling2_ctrl_up(const inkling2_ctrl_batch bt, const int n, const int r) {
    constexpr int v     = vec ? 16/sizeof(T) : 1;
    constexpr int lanes = INKLING2_CTRL_UP_LANES;

    const inkling2_ctrl_args & p = bt.a[blockIdx.z];
    const T * gain   = (const T *) p.gain;
    const T * forget = (const T *) p.forget;

    extern __shared__ float g_s[];

    const int t = blockIdx.y;
    for (int k = threadIdx.x; k < r; k += blockDim.x) {
        g_s[k] = p.g[t*r + k];
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
            inkling2_load<v>(forget + j*p.s_forget + k, wf);
            inkling2_load<v>(gain   + j*p.s_gain   + k, wa);
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
        const float c = p.cur[t*p.s_cur + j];
        const float m = p.msg[t*p.s_msg + j];
        p.dst[t*p.s_dst + j] = (c - c*tanhf(f)) + m*tanhf(a);
    }
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

// run n_ops ctrl ops of the same shape and weight types in one down and one up launch
static void inkling2_ctrl_run(ggml_backend_cuda_context & ctx, ggml_tensor * const * ops, const int n_ops) {
    const ggml_tensor * op0   = ops[0];
    const ggml_type     tdown = op0->src[2]->type;
    const ggml_type     tup   = op0->src[3]->type;
    const int           n     = op0->src[0]->ne[0];
    const int           r     = op0->src[2]->ne[1];
    const int64_t       nt    = op0->src[0]->ne[1];

    float eps;
    memcpy(&eps, op0->op_params, sizeof(float));

    ggml_cuda_pool_alloc<float> g(ctx.pool(), (size_t) n_ops*r*nt);

    inkling2_ctrl_batch bt = {};
    bool vec_down = n % (16/ggml_type_size(tdown)) == 0;
    bool vec_up   = r % (16/ggml_type_size(tup)) == 0;
    for (int o = 0; o < n_ops; ++o) {
        const ggml_tensor * cur    = ops[o]->src[0];
        const ggml_tensor * msg    = ops[o]->src[1];
        const ggml_tensor * down   = ops[o]->src[2];
        const ggml_tensor * gain   = ops[o]->src[3];
        const ggml_tensor * forget = ops[o]->src[4];
        inkling2_ctrl_args & p = bt.a[o];
        p.cur      = (const float *) cur->data;
        p.msg      = (const float *) msg->data;
        p.down     = down->data;
        p.gain     = gain->data;
        p.forget   = forget->data;
        p.g        = g.get() + (size_t) o*r*nt;
        p.dst      = (float *) ops[o]->data;
        p.s_cur    = cur->nb[1]/sizeof(float);
        p.s_msg    = msg->nb[1]/sizeof(float);
        p.s_down   = down->nb[1]/ggml_type_size(tdown);
        p.s_gain   = gain->nb[1]/ggml_type_size(tup);
        p.s_forget = forget->nb[1]/ggml_type_size(tup);
        p.s_dst    = ops[o]->nb[1]/sizeof(float);
        vec_down   = vec_down && inkling2_aligned(cur->data, cur->nb[1]) && inkling2_aligned(msg->data, msg->nb[1]) &&
            inkling2_aligned(down->data, down->nb[1]);
        vec_up     = vec_up && inkling2_aligned(gain->data, gain->nb[1]) && inkling2_aligned(forget->data, forget->nb[1]);
    }

    cudaStream_t stream = ctx.stream();

    auto down_t = [&](auto tag) {
        using T = decltype(tag);
        auto kernel = vec_down ? inkling2_ctrl_down<T, true> : inkling2_ctrl_down<T, false>;
        kernel<<<dim3(r, nt, n_ops), 256, 0, stream>>>(bt, n, r, eps);
    };
    switch (tdown) {
        case GGML_TYPE_F32:  down_t(float());       break;
        case GGML_TYPE_F16:  down_t(half());        break;
        case GGML_TYPE_BF16: down_t(nv_bfloat16()); break;
        default: GGML_ABORT("unsupported type %s", ggml_type_name(tdown));
    }

    const int n_threads = 256;
    auto up_t = [&](auto tag) {
        using T = decltype(tag);
        auto kernel = vec_up ? inkling2_ctrl_up<T, true> : inkling2_ctrl_up<T, false>;
        const dim3 blocks((n + n_threads/INKLING2_CTRL_UP_LANES - 1)/(n_threads/INKLING2_CTRL_UP_LANES), nt, n_ops);
        kernel<<<blocks, n_threads, r*sizeof(float), stream>>>(bt, n, r);
    };
    switch (tup) {
        case GGML_TYPE_F32:  up_t(float());       break;
        case GGML_TYPE_F16:  up_t(half());        break;
        case GGML_TYPE_BF16: up_t(nv_bfloat16()); break;
        default: GGML_ABORT("unsupported type %s", ggml_type_name(tup));
    }
}

void ggml_cuda_op_inkling2_ctrl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    inkling2_ctrl_run(ctx, &dst, 1);
}

static bool inkling2_ctrl_batchable(const ggml_tensor * a, const ggml_tensor * b) {
    if (b->op != GGML_OP_INKLING2_CTRL || !ggml_are_same_shape(a, b) || a->src[2]->type != b->src[2]->type ||
        a->src[3]->type != b->src[3]->type || a->src[2]->ne[1] != b->src[2]->ne[1] || a->src[3]->ne[0] != b->src[3]->ne[0] ||
        ggml_get_op_params_f32(a, 0) != ggml_get_op_params_f32(b, 0)) {
        return false;
    }
    const ggml_tensor * root_a = a->view_src ? a->view_src : a;
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        const ggml_tensor * src = b->src[s];
        if (src && (src->view_src ? src->view_src : src) == root_a) {
            return false;
        }
    }
    return true;
}

static bool inkling2_noop_node(const ggml_tensor * n) {
    return ggml_is_empty(n) || n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_PERMUTE ||
        n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_NONE;
}

// before allocation: the last node of a run of independent ctrl ops starting at node_idx, or -1
int ggml_cuda_inkling2_ctrl_batch_group(const ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_INKLING2_CTRL) {
        return -1;
    }
    std::vector<const ggml_tensor *> ops = { first };
    int last = -1;
    for (int j = node_idx + 1; j < cgraph->n_nodes && (int) ops.size() < INKLING2_CTRL_MAX; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (inkling2_noop_node(n)) {
            continue;
        }
        bool ok = true;
        for (const ggml_tensor * o : ops) {
            ok = ok && inkling2_ctrl_batchable(o, n);
        }
        if (!ok) {
            break;
        }
        ops.push_back(n);
        last = j;
    }
    return last;
}

int ggml_cuda_inkling2_ctrl_batch_fusion(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    const int last = ggml_cuda_inkling2_ctrl_batch_group(cgraph, node_idx);
    if (last < 0) {
        return 0;
    }
    std::vector<ggml_tensor *> ops;
    for (int j = node_idx; j <= last; ++j) {
        if (!inkling2_noop_node(cgraph->nodes[j])) {
            ops.push_back(cgraph->nodes[j]);
        }
    }
    // the ops run concurrently: no output may overlap an input or output of another op
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const uintptr_t a0 = (uintptr_t) a->data, a1 = a0 + ggml_nbytes(a);
        const uintptr_t b0 = (uintptr_t) b->data, b1 = b0 + ggml_nbytes(b);
        return a->data && b->data && a0 < b1 && b0 < a1;
    };
    for (size_t a = 0; a < ops.size(); ++a) {
        for (size_t b = 0; b < ops.size(); ++b) {
            if (a == b) {
                continue;
            }
            if (overlap(ops[a], ops[b])) {
                return 0;
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (ops[b]->src[s] && overlap(ops[a], ops[b]->src[s])) {
                    return 0;
                }
            }
        }
    }
    inkling2_ctrl_run(ctx, ops.data(), (int) ops.size());
    return last - node_idx;
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

struct inkling2_conv_args {
    const float   * cur;
    const float   * past;
    const float   * weight;
    const float   * dynamic;
    const float   * residual;
    const int32_t * ids;
    float         * y;
    float         * state;
    int             C, K, n_st, n_cph, offset;
    int64_t         s_cur1, s_past0, s_past1, s_past2, s_dyn1, s_res1, s_state_q;
    float           scale;
    int             post, hd;
    float           post_param;
};

struct inkling2_clip_args {
    inkling2_clip_src a;
    inkling2_clip_src b;
    const float     * tau;
    int               s_tau2;
    bool              has_scale;
};

static inkling2_conv_args inkling2_conv_make_args(const ggml_tensor * dst, float * state_dst, int64_t s_state_q) {
    const ggml_tensor * cur      = dst->src[0];
    const ggml_tensor * past     = dst->src[1];
    const ggml_tensor * weight   = dst->src[2];
    const ggml_tensor * dynamic  = dst->src[3];
    const ggml_tensor * residual = dst->src[4];
    const ggml_tensor * ids      = dst->src[5];

    inkling2_conv_args a = {};
    memcpy(&a.scale, dst->op_params, sizeof(float));
    a.offset     = ggml_get_op_params_i32(dst, 1);
    a.post       = ggml_get_op_params_i32(dst, 2);
    a.hd         = ggml_get_op_params_i32(dst, 3);
    a.post_param = ggml_get_op_params_f32(dst, 4);

    a.C      = cur->ne[0];
    a.K      = weight->ne[0];
    const int T      = cur->ne[1];
    const int n_seqs = ids ? ids->ne[0] : past->ne[2];
    a.n_st   = T/n_seqs;
    a.n_cph  = dynamic ? a.C/(dynamic->ne[0]/a.K) : 1;

    a.cur       = (const float *) cur->data;
    a.past      = (const float *) past->data;
    a.weight    = (const float *) weight->data;
    a.dynamic   = dynamic ? (const float *) dynamic->data : nullptr;
    a.residual  = residual ? (const float *) residual->data : nullptr;
    a.ids       = ids ? (const int32_t *) ids->data : nullptr;
    a.y         = (float *) dst->data;
    a.state     = state_dst ? state_dst : a.y + (int64_t) a.C*T;
    a.s_cur1    = cur->nb[1]/sizeof(float);
    a.s_past0   = past->nb[0]/sizeof(float);
    a.s_past1   = past->nb[1]/sizeof(float);
    a.s_past2   = ids ? 0 : past->nb[2]/sizeof(float);
    a.s_dyn1    = dynamic ? dynamic->nb[1]/sizeof(float) : 0;
    a.s_res1    = residual ? residual->nb[1]/sizeof(float) : 0;
    a.s_state_q = state_dst ? s_state_q : (int64_t) a.C*(a.K - 1);
    return a;
}

static __device__ __forceinline__ void inkling2_conv_block(const inkling2_conv_args & a, const int bx) {
    const int  c_raw = bx*blockDim.x + threadIdx.x;
    const bool valid = c_raw < a.C;
    const int  c     = valid ? c_raw : a.C - 1;

    const float * xp = a.ids ? a.past + a.ids[0]*a.s_past1 + (int64_t) (a.offset + c)*(a.K - 1) : a.past + c*a.s_past1;
    const float * xc = a.cur + c;
    const float * w  = a.weight + (int64_t) c*a.K;

    auto x = [&](int k) {
        return k < a.K - 1 ? xp[k*a.s_past0] : xc[(k - (a.K - 1))*a.s_cur1];
    };

    for (int i = 0; i < a.n_st; ++i) {
        float sum = 0.0f;
        for (int k = 0; k < a.K; ++k) {
            sum += x(i + k)*w[k];
        }

        if (a.dynamic) {
            const float * taps = a.dynamic + (int64_t) i*a.s_dyn1 + (c/a.n_cph)*a.K;
            for (int lag = 0; lag < a.K; ++lag) {
                sum += x(i + a.K - 1 - lag)*taps[lag];
            }
        }

        const float out = __fmul_rn(sum + x(i + a.K - 1), a.scale);
        float       yv  = a.residual ? a.residual[(int64_t) i*a.s_res1 + c] + out : out;

        if (a.post != GGML_INKLING2_HEAD_POST_NONE) {
            __shared__ float hs[1024/WARP_SIZE];
            const float ss = warp_reduce_sum(yv*yv);
            if (threadIdx.x % WARP_SIZE == 0) {
                hs[threadIdx.x / WARP_SIZE] = ss;
            }
            __syncthreads();
            float sum_h = 0.0f;
            const int w0 = (threadIdx.x / a.hd)*(a.hd / WARP_SIZE);
            for (int wi = 0; wi < a.hd / WARP_SIZE; ++wi) {
                sum_h += hs[w0 + wi];
            }
            __syncthreads();
            if (a.post == GGML_INKLING2_HEAD_POST_CLIP) {
                yv = yv/fmaxf(sqrtf(sum_h)*(1.0f/a.post_param) + 1e-6f/a.post_param, 1.0f);
            } else {
                yv = yv*rsqrtf(sum_h/a.hd + a.post_param);
            }
        }

        if (valid) {
            a.y[(int64_t) i*a.C + c] = yv;
        }
    }

    if (valid) {
        for (int k = 0; k < a.K - 1; ++k) {
            a.state[(int64_t) c*(a.K - 1) + k] = x(a.n_st + k);
        }
    }
}

static __global__ void inkling2_conv2_clip_f32(
        const inkling2_conv_args c0, const inkling2_conv_args c1, const inkling2_clip_args cl,
        const int b0, const int b1) {
    const int bx = blockIdx.x;
    if (bx < b0) {
        inkling2_conv_block(c0, bx);
        return;
    }
    if (bx < b0 + b1) {
        inkling2_conv_block(c1, bx - b0);
        return;
    }

    int64_t ir = (int64_t) (bx - b0 - b1)*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    if (ir >= cl.a.nr + cl.b.nr) {
        return;
    }
    const inkling2_clip_src & p = ir < cl.a.nr ? cl.a : cl.b;
    if (ir >= cl.a.nr) {
        ir -= cl.a.nr;
    }
    const int lane = threadIdx.x % WARP_SIZE;

    const int64_t i1 = ir % p.ne01;
    const int64_t i2 = (ir/p.ne01) % p.ne02;
    const int64_t i3 = ir/((int64_t) p.ne01*p.ne02);

    const float * xs = p.x + i1*p.s01 + i2*p.s02 + i3*p.s03;
    const float * ss = cl.has_scale ? p.scale + i1*p.s_scale1 : nullptr;
    float       * y  = p.dst + ir*p.ne00;

    float sum = 0.0f;
    for (int i0 = lane; i0 < p.ne00; i0 += WARP_SIZE) {
        const float v = cl.has_scale ? xs[i0]*ss[i0] : xs[i0];
        sum += v*v;
    }
    sum = warp_reduce_sum(sum);

    const float d = fmaxf(sqrtf(sum)*p.inv_threshold + p.eps_threshold, 1.0f);
    const float t = cl.tau ? cl.tau[i2*cl.s_tau2] : 1.0f;

    for (int i0 = lane; i0 < p.ne00; i0 += WARP_SIZE) {
        const float v = cl.has_scale ? xs[i0]*ss[i0] : xs[i0];
        y[i0] = v/d*t;
    }
}

static bool inkling2_tensor_depends_on(const ggml_tensor * consumer, const ggml_tensor * producer) {
    const ggml_tensor * prod_root = producer->view_src ? producer->view_src : producer;
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        const ggml_tensor * src = consumer->src[s];
        if (!src) {
            continue;
        }
        const ggml_tensor * src_root = src->view_src ? src->view_src : src;
        if (src_root == prod_root || src->data == producer->data) {
            return true;
        }
    }
    return false;
}

static bool inkling2_ranges_overlap(const void * a, size_t na, const void * b, size_t nb) {
    if (!a || !b || na == 0 || nb == 0) {
        return false;
    }
    const uintptr_t a0 = (uintptr_t) a;
    const uintptr_t a1 = a0 + na;
    const uintptr_t b0 = (uintptr_t) b;
    const uintptr_t b1 = b0 + nb;
    return a0 < b1 && b0 < a1;
}

// before allocation: the last node of a conv, conv, (clip) group that ggml_cuda_inkling2_conv_batch_fusion may run as one
// launch, or -1. The caller keeps the inputs of the group alive until that node, so that they do not alias its outputs.
int ggml_cuda_inkling2_conv_batch_group(const ggml_cgraph * cgraph, int node_idx) {
    auto single_chunk = [](const ggml_tensor * conv) {
        const ggml_tensor * ids    = conv->src[5];
        const int64_t       T      = conv->src[0]->ne[1];
        const int64_t       n_seqs = ids ? ids->ne[0] : conv->src[1]->ne[2];
        return n_seqs == 1 && T <= INKLING2_CONV_CHUNK;
    };
    auto next = [&](int start) {
        for (int j = start; j < cgraph->n_nodes; ++j) {
            const ggml_tensor * n = cgraph->nodes[j];
            if (ggml_is_empty(n) || n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_PERMUTE ||
                n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_NONE || n->op == GGML_OP_CPY) {
                continue;
            }
            return j;
        }
        return -1;
    };
    // no data pointers yet: compare the root tensors only
    auto depends = [](const ggml_tensor * consumer, const ggml_tensor * producer) {
        const ggml_tensor * prod_root = producer->view_src ? producer->view_src : producer;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = consumer->src[s];
            if (src && (src->view_src ? src->view_src : src) == prod_root) {
                return true;
            }
        }
        return false;
    };
    const ggml_tensor * conv0 = cgraph->nodes[node_idx];
    if (conv0->op != GGML_OP_INKLING2_CONV || !single_chunk(conv0)) {
        return -1;
    }
    const int idx1 = next(node_idx + 1);
    if (idx1 < 0 || cgraph->nodes[idx1]->op != GGML_OP_INKLING2_CONV) {
        return -1;
    }
    const ggml_tensor * conv1 = cgraph->nodes[idx1];
    if (!single_chunk(conv1) || depends(conv1, conv0)) {
        return -1;
    }
    const int idx2 = next(idx1 + 1);
    if (idx2 >= 0 && cgraph->nodes[idx2]->op == GGML_OP_INKLING2_CLIP &&
        !depends(cgraph->nodes[idx2], conv0) && !depends(cgraph->nodes[idx2], conv1)) {
        return idx2;
    }
    return idx1;
}

int ggml_cuda_inkling2_conv_batch_fusion(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    ggml_tensor * conv0 = cgraph->nodes[node_idx];
    if (conv0->op != GGML_OP_INKLING2_CONV) {
        return 0;
    }

    auto conv_is_single_chunk = [](const ggml_tensor * conv) {
        const ggml_tensor * ids    = conv->src[5];
        const int64_t       T      = conv->src[0]->ne[1];
        const int64_t       n_seqs = ids ? ids->ne[0] : conv->src[1]->ne[2];
        return n_seqs == 1 && T <= INKLING2_CONV_CHUNK;
    };
    if (!conv_is_single_chunk(conv0)) {
        return 0;
    }

    float * state_dst0 = nullptr;
    int64_t s_state_q0 = 0;
    const int skip0 = ggml_cuda_inkling2_conv_state_fusion(cgraph, node_idx, &state_dst0, &s_state_q0);

    auto next_compute = [&](int start) {
        for (int j = start; j < cgraph->n_nodes; ++j) {
            const ggml_tensor * n = cgraph->nodes[j];
            if (ggml_is_empty(n) || n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE ||
                n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_NONE) {
                continue;
            }
            return j;
        }
        return -1;
    };

    const int idx1 = next_compute(node_idx + skip0 + 1);
    if (idx1 < 0 || cgraph->nodes[idx1]->op != GGML_OP_INKLING2_CONV) {
        return 0;
    }
    ggml_tensor * conv1 = cgraph->nodes[idx1];
    if (!conv_is_single_chunk(conv1) || inkling2_tensor_depends_on(conv1, conv0)) {
        return 0;
    }

    float * state_dst1 = nullptr;
    int64_t s_state_q1 = 0;
    const int skip1 = ggml_cuda_inkling2_conv_state_fusion(cgraph, idx1, &state_dst1, &s_state_q1);

    // check that conv0 and conv1 touch disjoint channel ranges if they share a recurrent state buffer
    const int64_t C0   = conv0->src[0]->ne[0];
    const int64_t C1   = conv1->src[0]->ne[0];
    const int64_t K0   = conv0->src[2]->ne[0];
    const int64_t K1   = conv1->src[2]->ne[0];
    const int     off0 = ggml_get_op_params_i32(conv0, 1);
    const int     off1 = ggml_get_op_params_i32(conv1, 1);

    if (conv0->src[5] && conv1->src[5]) {
        const ggml_tensor * p0 = conv0->src[1]->view_src ? conv0->src[1]->view_src : conv0->src[1];
        const ggml_tensor * p1 = conv1->src[1]->view_src ? conv1->src[1]->view_src : conv1->src[1];
        if (p0 == p1 && off0 < off1 + C1 && off1 < off0 + C0) {
            return 0;
        }
    }
    if (state_dst0 && state_dst1 &&
        inkling2_ranges_overlap(state_dst0, (size_t) (K0 - 1)*C0*sizeof(float),
                                state_dst1, (size_t) (K1 - 1)*C1*sizeof(float))) {
        return 0;
    }
    if (state_dst0 && inkling2_ranges_overlap(state_dst0, (size_t) (K0 - 1)*C0*sizeof(float),
                                              conv1->src[1]->data, ggml_nbytes(conv1->src[1]))) {
        // state_dst0 is inside conv1's past buffer: safe only when indexed by ids with disjoint channel offsets
        if (!(conv0->src[5] && conv1->src[5] && (off0 + C0 <= off1 || off1 + C1 <= off0))) {
            return 0;
        }
    }
    if (state_dst1 && inkling2_ranges_overlap(state_dst1, (size_t) (K1 - 1)*C1*sizeof(float),
                                              conv0->src[1]->data, ggml_nbytes(conv0->src[1]))) {
        if (!(conv0->src[5] && conv1->src[5] && (off0 + C0 <= off1 || off1 + C1 <= off0))) {
            return 0;
        }
    }

    int last_idx = idx1 + skip1;
    inkling2_clip_args cl = {};
    int b2 = 0;

    const int idx2 = next_compute(idx1 + skip1 + 1);
    if (idx2 >= 0 && cgraph->nodes[idx2]->op == GGML_OP_INKLING2_CLIP) {
        ggml_tensor * clip = cgraph->nodes[idx2];
        if (ggml_is_contiguous(clip) &&
            !inkling2_tensor_depends_on(clip, conv0) &&
            !inkling2_tensor_depends_on(clip, conv1)) {
            const ggml_tensor * x      = clip->src[0];
            const ggml_tensor * scale  = clip->src[1];
            const ggml_tensor * tau    = clip->src[2];
            const ggml_tensor * x2     = clip->src[3];
            const ggml_tensor * scale2 = clip->src[4];
            if (!x2 || !scale == !scale2) {
                const float threshold  = ggml_get_op_params_f32(clip, 0);
                const float eps        = ggml_get_op_params_f32(clip, 1);
                const float threshold2 = ggml_get_op_params_f32(clip, 2);
                float * y = (float *) clip->data;
                cl.a = inkling2_clip_make_src(x, scale, y, threshold, eps);
                if (x2) {
                    cl.b = inkling2_clip_make_src(x2, scale2, y + ggml_nelements(x), threshold2, eps);
                }
                cl.tau       = tau ? (const float *) tau->data : nullptr;
                cl.s_tau2    = tau && tau->ne[2] > 1 ? 1 : 0;
                cl.has_scale = scale != nullptr;

                const int rows_per_block = 256/WARP_SIZE;
                b2 = (int) ((cl.a.nr + cl.b.nr + rows_per_block - 1)/rows_per_block);
                last_idx = idx2;
            }
        }
    }

    // the fused ops run concurrently, but the allocator assumed graph order: it may have placed the output of
    // a later op over an input of an earlier op whose last use was that op. Every write range of one op must be
    // disjoint from every read and write range of the others (the shared conv state cache is checked above).
    {
        struct range { const void * p; size_t n; };
        auto reads = [](const ggml_tensor * t, range * r) {
            int n = 0;
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (t->src[s] && t->src[s]->data) {
                    r[n++] = { t->src[s]->data, ggml_nbytes(t->src[s]) };
                }
            }
            return n;
        };
        const ggml_tensor * clip = b2 > 0 ? cgraph->nodes[idx2] : nullptr;
        const ggml_tensor * ops[3] = { conv0, conv1, clip };
        range wr[3][2] = {
            { { conv0->data, ggml_nbytes(conv0) }, { state_dst0, state_dst0 ? (size_t) (K0 - 1)*C0*sizeof(float) : 0 } },
            { { conv1->data, ggml_nbytes(conv1) }, { state_dst1, state_dst1 ? (size_t) (K1 - 1)*C1*sizeof(float) : 0 } },
            { { clip ? clip->data : nullptr, clip ? ggml_nbytes(clip) : 0 }, { nullptr, 0 } },
        };
        for (int x = 0; x < 3; ++x) {
            for (int y = 0; y < 3; ++y) {
                if (x == y || !ops[x] || !ops[y]) {
                    continue;
                }
                range rd[GGML_MAX_SRC + 2];
                int nr = reads(ops[y], rd);
                rd[nr++] = wr[y][0];
                rd[nr++] = wr[y][1];
                for (int w = 0; w < 2; ++w) {
                    for (int k = 0; k < nr; ++k) {
                        if (!inkling2_ranges_overlap(wr[x][w].p, wr[x][w].n, rd[k].p, rd[k].n)) {
                            continue;
                        }
                        // the conv state stores into the shared cache, already checked for disjoint channels
                        const bool state_vs_past = x < 2 && y < 2 && w == 1 &&
                            (k == nr - 1 || rd[k].p == ops[y]->src[1]->data) &&
                            conv0->src[5] && conv1->src[5] && (off0 + C0 <= off1 || off1 + C1 <= off0);
                        if (!state_vs_past) {
                            return 0;
                        }
                    }
                }
            }
        }
    }

    const inkling2_conv_args a0 = inkling2_conv_make_args(conv0, state_dst0, s_state_q0);
    const inkling2_conv_args a1 = inkling2_conv_make_args(conv1, state_dst1, s_state_q1);
    const int n_threads = 256;
    const int b0 = (a0.C + n_threads - 1)/n_threads;
    const int b1 = (a1.C + n_threads - 1)/n_threads;
    GGML_ASSERT(a0.post == GGML_INKLING2_HEAD_POST_NONE || (n_threads % a0.hd == 0 && a0.hd % WARP_SIZE == 0));
    GGML_ASSERT(a1.post == GGML_INKLING2_HEAD_POST_NONE || (n_threads % a1.hd == 0 && a1.hd % WARP_SIZE == 0));

    inkling2_conv2_clip_f32<<<dim3(b0 + b1 + b2), n_threads, 0, ctx.stream()>>>(a0, a1, cl, b0, b1);
    return last_idx - node_idx;
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

// single-token conv of channel c (same arithmetic as inkling2_conv_f32 with one token and no head post op):
// writes y[c] and the new state of channel c, returns y[c]
static __device__ __forceinline__ float inkling2_conv1(const inkling2_conv_args & a, const int c) {
    const int     K  = a.K;
    const float * xp = a.ids ? a.past + a.ids[0]*a.s_past1 + (int64_t) (a.offset + c)*(K - 1) : a.past + c*a.s_past1;
    const float * xc = a.cur + c;
    const float * w  = a.weight + (int64_t) c*K;

    auto x = [&](int k) {
        return k < K - 1 ? xp[k*a.s_past0] : xc[(k - (K - 1))*a.s_cur1];
    };

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += x(k)*w[k];
    }
    if (a.dynamic) {
        const float * taps = a.dynamic + (c/a.n_cph)*K;
        for (int lag = 0; lag < K; ++lag) {
            sum += x(K - 1 - lag)*taps[lag];
        }
    }

    // inkling2_conv_f32 compiles residual + __fmul_rn(.., scale) to one fma on HIP: match it bit for bit
    const float yv = a.residual ? fmaf(sum + x(K - 1), a.scale, a.residual[c]) : __fmul_rn(sum + x(K - 1), a.scale);

    a.y[c] = yv;
    for (int k = 0; k < K - 1; ++k) {
        a.state[(int64_t) c*(K - 1) + k] = x(1 + k);
    }
    return yv;
}

// one token: conv of all channels, then the depth mix of its output with one query, in one block.
// All conv reads happen before the first barrier and all mix writes after the last one, so the mix
// output may alias the conv inputs.
extern __shared__ float inkling2_conv_mix_ys[];

template <int NB>
static __global__ void inkling2_conv_mix_f32(
        const inkling2_conv_args a, const float * __restrict__ query, const float * __restrict__ blocks, float * __restrict__ dst,
        const int n, const int64_t s_b1, const float scale, const float eps) {
    const int n4 = n / 4;
    const int i4 = threadIdx.x;

    float4 xv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 bv[NB > 0 ? NB : 1];
    float  dot[NB > 0 ? NB : 1];
#pragma unroll
    for (int j = 0; j < NB; ++j) {
        dot[j] = 0.0f;
    }
    float ss = 0.0f;
    float dx = 0.0f;

    // K = 4 without dynamic taps and 16-byte aligned conv weights: the coalesced conv path
    const bool fast = a.K == 4 && !a.dynamic && ((uintptr_t) a.weight % 16) == 0;

    if (i4 < n4) {
        if (fast) {
            // channel c = i4 + j*n4 per thread so that the loads of a warp are contiguous; y goes through LDS to
            // the float4 layout of the mix. Same arithmetic as inkling2_conv1.
            float * ys = inkling2_conv_mix_ys;
            const float * past = a.ids ? a.past + a.ids[0]*a.s_past1 + (int64_t) a.offset*3 : a.past;
            const int64_t s_pc = a.ids ? 3 : a.s_past1;
            float xp[4][3], xc[4], rv[4];
            float4 w[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = i4 + j*n4;
#pragma unroll
                for (int k = 0; k < 3; ++k) {
                    xp[j][k] = past[c*s_pc + k*a.s_past0];
                }
                xc[j] = a.cur[c];
                w[j]  = ((const float4 *) a.weight)[c];
                rv[j] = a.residual ? a.residual[c] : 0.0f;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = i4 + j*n4;
                float sum = 0.0f;
                sum += xp[j][0]*w[j].x;
                sum += xp[j][1]*w[j].y;
                sum += xp[j][2]*w[j].z;
                sum += xc[j]*w[j].w;
                const float yv = a.residual ? fmaf(sum + xc[j], a.scale, rv[j]) : __fmul_rn(sum + xc[j], a.scale);
                a.y[c] = yv;
                ys[c]  = yv;
                a.state[(int64_t) c*3 + 0] = xp[j][1];
                a.state[(int64_t) c*3 + 1] = xp[j][2];
                a.state[(int64_t) c*3 + 2] = xc[j];
            }
        }
    }
    if (fast) {
        __syncthreads();
    }
    if (i4 < n4) {
        if (fast) {
            xv = ((const float4 *) inkling2_conv_mix_ys)[i4];
        } else {
            xv.x = inkling2_conv1(a, 4*i4 + 0);
            xv.y = inkling2_conv1(a, 4*i4 + 1);
            xv.z = inkling2_conv1(a, 4*i4 + 2);
            xv.w = inkling2_conv1(a, 4*i4 + 3);
        }
        const float4 qv = ((const float4 *) query)[i4];
        ss = xv.x*xv.x + xv.y*xv.y + xv.z*xv.z + xv.w*xv.w;
        dx = xv.x*qv.x + xv.y*qv.y + xv.z*qv.z + xv.w*qv.w;
#pragma unroll
        for (int j = 0; j < NB; ++j) {
            const float4 b = ((const float4 *) (blocks + j*s_b1))[i4];
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
        ((float4 *) dst)[i4] = acc;
    }
}

static int inkling2_next_compute(const ggml_cgraph * cgraph, int start) {
    for (int j = start; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_is_empty(n) || n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE ||
            n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_NONE) {
            continue;
        }
        return j;
    }
    return -1;
}

int ggml_cuda_inkling2_conv_mix_fusion(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    ggml_tensor * conv = cgraph->nodes[node_idx];
    if (conv->op != GGML_OP_INKLING2_CONV || (conv->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }
    const ggml_tensor * ids    = conv->src[5];
    const int64_t       C      = conv->src[0]->ne[0];
    const int64_t       T      = conv->src[0]->ne[1];
    const int64_t       n_seqs = ids ? ids->ne[0] : conv->src[1]->ne[2];
    if (T != 1 || n_seqs != 1 || ggml_get_op_params_i32(conv, 2) != GGML_INKLING2_HEAD_POST_NONE ||
        C % 4 != 0 || C/4 > 1024) {
        return 0;
    }

    float * state_dst = nullptr;
    int64_t s_state_q = 0;
    const int skip = ggml_cuda_inkling2_conv_state_fusion(cgraph, node_idx, &state_dst, &s_state_q);

    const int idx_mix = inkling2_next_compute(cgraph, node_idx + skip + 1);
    if (idx_mix < 0) {
        return 0;
    }
    ggml_tensor * mix = cgraph->nodes[idx_mix];
    if (mix->op != GGML_OP_INKLING2_MIX) {
        return 0;
    }
    const ggml_tensor * x      = mix->src[0];
    const ggml_tensor * query  = mix->src[1];
    const ggml_tensor * blocks = mix->src[2];
    const int n_blocks = blocks ? blocks->ne[1] : 0;
    if (x->data != conv->data || x->ne[0] != C || x->ne[1] != 1 || query->type != GGML_TYPE_F32 || query->ne[1] != 1 ||
        n_blocks > 6 || !inkling2_aligned(conv->data, 16) || !inkling2_aligned(mix->data, 16) ||
        !inkling2_aligned(query->data, 16) || (blocks && !inkling2_aligned(blocks->data, blocks->nb[1]))) {
        return 0;
    }

    float scale;
    float eps;
    memcpy(&scale, (const float *) mix->op_params + 0, sizeof(float));
    memcpy(&eps,   (const float *) mix->op_params + 1, sizeof(float));

    const inkling2_conv_args a = inkling2_conv_make_args(conv, state_dst, s_state_q);
    const float * b_d  = blocks ? (const float *) blocks->data : nullptr;
    const int64_t s_b1 = blocks ? blocks->nb[1]/sizeof(float) : 0;
    const int n_threads = (int) ((C/4 + 63)/64)*64;

    auto go = [&](auto nb_tag) {
        constexpr int NB = decltype(nb_tag)::value;
        inkling2_conv_mix_f32<NB><<<1, n_threads, C*sizeof(float), ctx.stream()>>>(a, (const float *) query->data, b_d, (float *) mix->data,
                (int) C, s_b1, scale, eps);
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
    return idx_mix - node_idx;
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
template <typename T, typename M, typename D>
static __global__ void inkling2_rel_bias_f32(
        const T * __restrict__ table, const float * __restrict__ r, const float * __restrict__ sinks, const int32_t * __restrict__ rel_pos,
        const M * __restrict__ mask, D * __restrict__ dst, const int n_rel, const int n_extent, const int n_kv, const int n_ps,
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

    const int64_t dst_idx = (t % n_ps)*s_dst1 + h*s_dst2 + (t / n_ps)*s_dst3 + kv;

    float mv = 0.0f;
    if constexpr (!std::is_same_v<M, void>) {
        mv = ggml_cuda_cast<float>(mask[(t % n_ps)*s_mask1 + (t / n_ps)*s_mask3 + kv]);
        if (mv <= -1e30f) {
            dst[dst_idx] = ggml_cuda_cast<D>(mv);
            return;
        }
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
        v += mv;
    }

    dst[dst_idx] = ggml_cuda_cast<D>(v);
}

bool ggml_cuda_inkling2_rel_bias_supported(const ggml_tensor * op) {
    const ggml_type type = op->src[0]->type;
    return (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) &&
        (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
        op->src[1]->ne[1] <= 65535 && op->src[1]->ne[2] <= 65535;
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

    const size_t dst_ts = ggml_type_size(dst->type);

    auto launch = [&](auto kernel, const auto * table_d, const auto * mask_d, auto * dst_d) {
        kernel<<<grid, n_threads, smem, ctx.stream()>>>(table_d, (const float *) r->data, (const float *) sinks->data,
                (const int32_t *) rel_pos->data, mask_d, dst_d, n_rel, n_extent, n_kv, n_ps, s_table1,
                r->nb[1]/sizeof(float), r->nb[2]/sizeof(float), sinks->nb[1]/sizeof(float), rel_pos->nb[1]/sizeof(int32_t),
                s_mask1, s_mask3, dst->nb[1]/dst_ts, dst->nb[2]/dst_ts, dst->nb[3]/dst_ts);
    };
    auto launch_dst = [&](auto t_tag, auto m_tag, const auto * table_d, const auto * mask_d) {
        using T = decltype(t_tag);
        using M = std::remove_pointer_t<decltype(m_tag)>;
        if (dst->type == GGML_TYPE_F16) {
            launch(inkling2_rel_bias_f32<T, M, half>, table_d, mask_d, (half *) dst->data);
        } else {
            launch(inkling2_rel_bias_f32<T, M, float>, table_d, mask_d, (float *) dst->data);
        }
    };
    auto launch_mask = [&](auto t_tag, const auto * table_d) {
        if (!mask) {
            launch_dst(t_tag, (void *) nullptr, table_d, (const void *) nullptr);
        } else if (mask->type == GGML_TYPE_F16) {
            launch_dst(t_tag, (half *) nullptr, table_d, (const half *) mask->data);
        } else {
            launch_dst(t_tag, (float *) nullptr, table_d, (const float *) mask->data);
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
    const void  * w[INKLING2_PROJ_MAX_W];
    int64_t       s_w[INKLING2_PROJ_MAX_W];       // row stride in elements
    int           rows[INKLING2_PROJ_MAX_W];
    int           out_off[INKLING2_PROJ_MAX_W];   // first output row in the dst of this weight
    int           blk_start[INKLING2_PROJ_MAX_W + 1];
    // input, input norm (or null) and output of each weight: several proj ops can run as one launch
    const float * x[INKLING2_PROJ_MAX_W];
    const float * norm[INKLING2_PROJ_MAX_W];
    float       * dst[INKLING2_PROJ_MAX_W];
    int64_t       s_x1[INKLING2_PROJ_MAX_W];
    int64_t       s_d1[INKLING2_PROJ_MAX_W];
    int           rpb;                            // rows per block
};

// the input row (normalized) in LDS, then one warp per nr weight rows in parallel
template <typename T, bool vec, bool has_norm>
static __global__ void inkling2_proj_f32(const inkling2_proj_weights p, const int n, const float eps) {
    constexpr int v  = vec ? 16/sizeof(T) : 1;
    constexpr int nw = 512 / WARP_SIZE;
    constexpr int nr = INKLING2_PROJ_ROWS_PER_BLOCK / nw;

    extern __shared__ float xs[];

    int k = 0;
    while ((int) blockIdx.x >= p.blk_start[k + 1]) {
        ++k;
    }

    const int t = blockIdx.y;
    const float * xr   = p.x[k] + t*p.s_x1[k];
    const float * norm = p.norm[k];
    float       * dst  = p.dst[k];
    const int64_t s_d1 = p.s_d1[k];

    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xi = xr[i];
        xs[inkling2_proj_xs_idx(i)] = xi;
        ss += xi*xi;
    }
    if constexpr (has_norm) {
        __shared__ float buf[32];
        ss = block_reduce<block_reduce_method::SUM>(ss, buf);
        if (norm) {
            const float scale = rsqrtf(ss/n + eps);
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                const int idx = inkling2_proj_xs_idx(i);
                xs[idx] = (xs[idx]*scale)*norm[i];
            }
        }
    }
    __syncthreads();

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

// add the weights of proj op `op` to p (weights [n_w0, n_w0 + n_w)); false when they do not fit
static bool inkling2_proj_add(inkling2_proj_weights & p, int & n_w_total, const ggml_tensor * op, bool & aligned) {
    const ggml_tensor * x    = op->src[0];
    const int           n_w  = ggml_get_op_params_i32(op, 1);
    const ggml_tensor * norm = op->src[1 + n_w];
    const size_t        ts   = ggml_type_size(op->src[1]->type);
    if (n_w_total + n_w > INKLING2_PROJ_MAX_W) {
        return false;
    }
    int off = 0;
    for (int i = 0; i < n_w; ++i) {
        const ggml_tensor * w = op->src[1 + i];
        const int           j = n_w_total + i;
        p.w[j]             = w->data;
        p.s_w[j]           = w->nb[1]/ts;
        p.rows[j]          = w->ne[1];
        p.out_off[j]       = off;
        p.blk_start[j + 1] = p.blk_start[j] + (w->ne[1] + p.rpb - 1)/p.rpb;
        p.x[j]             = (const float *) x->data;
        p.norm[j]          = norm ? (const float *) norm->data : nullptr;
        p.dst[j]           = (float *) op->data;
        p.s_x1[j]          = x->nb[1]/sizeof(float);
        p.s_d1[j]          = op->nb[1]/sizeof(float);
        off               += w->ne[1];
        aligned            = aligned && inkling2_aligned(w->data, w->nb[1]);
    }
    n_w_total += n_w;
    return true;
}

static void inkling2_proj_launch(ggml_backend_cuda_context & ctx, inkling2_proj_weights & p, const int n_w, const ggml_type type,
        const int n, const int64_t n_tokens, const bool any_norm, bool aligned, const float eps) {
    const size_t ts = ggml_type_size(type);
    aligned = aligned && n % (16/ts) == 0;
    for (int i = n_w; i < INKLING2_PROJ_MAX_W; ++i) {
        p.blk_start[i + 1] = p.blk_start[n_w];
    }

    const dim3   grid(p.blk_start[n_w], n_tokens);
    const int    n_threads = 512;
    const size_t smem      = inkling2_proj_xs_idx(n)*sizeof(float);

    auto launch = [&](auto kernel) {
        kernel<<<grid, n_threads, smem, ctx.stream()>>>(p, n, eps);
    };
    auto launch_t = [&](auto tag) {
        using T = decltype(tag);
        if (aligned) {
            any_norm ? launch(inkling2_proj_f32<T, true,  true>) : launch(inkling2_proj_f32<T, true,  false>);
        } else {
            any_norm ? launch(inkling2_proj_f32<T, false, true>) : launch(inkling2_proj_f32<T, false, false>);
        }
    };

    switch (type) {
        case GGML_TYPE_F32:  launch_t(float());       break;
        case GGML_TYPE_F16:  launch_t(half());        break;
        case GGML_TYPE_BF16: launch_t(nv_bfloat16()); break;
        default: GGML_ABORT("unsupported type %s", ggml_type_name(type));
    }
}

void ggml_cuda_op_inkling2_proj(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];

    float eps;
    memcpy(&eps, (const float *) dst->op_params + 0, sizeof(float));
    const int n_w = ggml_get_op_params_i32(dst, 1);

    // each block normalizes the input again, so a block takes many rows
    inkling2_proj_weights p = {};
    p.rpb = INKLING2_PROJ_ROWS_PER_BLOCK;
    bool aligned = true;
    int  n_w_total = 0;
    inkling2_proj_add(p, n_w_total, dst, aligned);
    inkling2_proj_launch(ctx, p, n_w_total, dst->src[1]->type, (int) x->ne[0], x->ne[1], dst->src[1 + n_w] != nullptr, aligned, eps);
}

static bool inkling2_is_noop(const ggml_tensor * n) {
    return ggml_is_empty(n) || n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_PERMUTE ||
        n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_NONE;
}

// proj ops that may run as one launch: same weight type, input width and eps, one token, independent of each other
static bool inkling2_proj_batchable(const ggml_tensor * a, const ggml_tensor * b) {
    if (b->op != GGML_OP_INKLING2_PROJ || a->src[1]->type != b->src[1]->type || a->src[0]->ne[0] != b->src[0]->ne[0] ||
        a->src[0]->ne[1] != 1 || b->src[0]->ne[1] != 1 || ggml_get_op_params_f32(a, 0) != ggml_get_op_params_f32(b, 0)) {
        return false;
    }
    const ggml_tensor * root_a = a->view_src ? a->view_src : a;
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        const ggml_tensor * src = b->src[s];
        if (src && (src->view_src ? src->view_src : src) == root_a) {
            return false;
        }
    }
    return true;
}

// before allocation: the last node of a run of batchable proj ops starting at node_idx, or -1
int ggml_cuda_inkling2_proj_batch_group(const ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_INKLING2_PROJ || first->src[0]->ne[1] != 1) {
        return -1;
    }
    int n_w  = ggml_get_op_params_i32(first, 1);
    int last = -1;
    std::vector<const ggml_tensor *> ops = { first };
    for (int j = node_idx + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (inkling2_is_noop(n)) {
            continue;
        }
        bool ok = n_w + ggml_get_op_params_i32(n, 1) <= INKLING2_PROJ_MAX_W;
        for (const ggml_tensor * o : ops) {
            ok = ok && inkling2_proj_batchable(o, n);
        }
        if (!ok) {
            break;
        }
        n_w += ggml_get_op_params_i32(n, 1);
        ops.push_back(n);
        last = j;
    }
    return last;
}

// run a group of proj ops (see ggml_cuda_inkling2_proj_batch_group) as one launch; returns nodes to skip or 0
int ggml_cuda_inkling2_proj_batch_fusion(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    const int last = ggml_cuda_inkling2_proj_batch_group(cgraph, node_idx);
    if (last < 0) {
        return 0;
    }
    std::vector<const ggml_tensor *> ops;
    for (int j = node_idx; j <= last; ++j) {
        if (!inkling2_is_noop(cgraph->nodes[j])) {
            ops.push_back(cgraph->nodes[j]);
        }
    }
    // the launch runs the ops concurrently: no output may overlap an input or output of another op
    for (size_t a = 0; a < ops.size(); ++a) {
        for (size_t b = 0; b < ops.size(); ++b) {
            if (a == b) {
                continue;
            }
            if (inkling2_ranges_overlap(ops[a]->data, ggml_nbytes(ops[a]), ops[b]->data, ggml_nbytes(ops[b]))) {
                return 0;
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const ggml_tensor * src = ops[b]->src[s];
                if (src && inkling2_ranges_overlap(ops[a]->data, ggml_nbytes(ops[a]), src->data, ggml_nbytes(src))) {
                    return 0;
                }
            }
        }
    }

    inkling2_proj_weights p = {};
    p.rpb = INKLING2_PROJ_ROWS_PER_BLOCK;
    bool aligned   = true;
    bool any_norm  = false;
    int  n_w_total = 0;
    for (const ggml_tensor * op : ops) {
        GGML_ASSERT(inkling2_proj_add(p, n_w_total, op, aligned));
        any_norm = any_norm || op->src[1 + ggml_get_op_params_i32(op, 1)] != nullptr;
    }
    inkling2_proj_launch(ctx, p, n_w_total, ops[0]->src[1]->type, (int) ops[0]->src[0]->ne[0], 1, any_norm, aligned,
            ggml_get_op_params_f32(ops[0], 0));
    return last - node_idx;
}

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
        const int   j      = j0 + threadIdx.x / lanes;
        const bool  valid  = j < n_kv;
        const float mv     = valid ? ggml_cuda_cast<float>(mask[j]) : -INFINITY;
        const bool  active = valid && mv > -1e30f;

        float part = 0.0f;
        if (active) {
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
            const float l = active ? ((part - sh) + mv) : -INFINITY;
            lg[j] = l;
            mx = fmaxf(mx, l);
        }
    }

    __shared__ float buf[32];
    mx = block_reduce<block_reduce_method::MAX>(mx, buf);
    __syncthreads();

    float sum = 0.0f;
    for (int j = threadIdx.x; j < n_kv; j += blockDim.x) {
        const float lv = lg[j];
        const float e  = lv > -1e30f ? expf(lv - mx) : 0.0f;
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

// Q8_0 KV cache (non-transposed V): 8 lanes per kv cell for K (16 elems/lane), coalesced d/4 threads per V row
template <typename M>
static __global__ void inkling2_attn_q8_0_f32(
        const float * __restrict__ q, const char * __restrict__ k, const char * __restrict__ v, const float * __restrict__ table,
        const float * __restrict__ r, const float * __restrict__ sinks, const int32_t * __restrict__ rel_pos, const M * __restrict__ mask,
        const float * __restrict__ gate, float * __restrict__ dst, const int d, const int n_gqa, const int n_kv, const int n_rel, const int n_extent,
        const int64_t s_q1, const int64_t s_k1, const int64_t s_k2, const int64_t s_v1, const int64_t s_v2, const int64_t s_t1, const int64_t s_r1,
        const float scale, const int n_rot_q, const int n_rot_v) {
    constexpr int lanes = 8;

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
    const int   nb_half = d / 16;

    // all lanes of a group take part in the shuffles, so the loop runs the same count in every group
    float mx = 0.0f; // the zero sink
    for (int j0 = 0; j0 < n_kv; j0 += ng) {
        const int   j      = j0 + threadIdx.x / lanes;
        const bool  valid  = j < n_kv;
        const float mv     = valid ? ggml_cuda_cast<float>(mask[j]) : -INFINITY;
        const bool  active = valid && mv > -1e30f;

        float part = 0.0f;
        if (active) {
            const block_q8_0 * kj = (const block_q8_0 *) (k + hk*s_k1 + j*s_k2);
            float qk = 0.0f;
            for (int b = li; b < nb_half; b += lanes) {
                const int ib   = b >> 1;
                const int qoff = (b & 1) * 16;
                const float kd = __half2float(kj[ib].d);
                const float * qp = qs + b*16;
                float dot = 0.0f;
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    int q4;
                    ggml_cuda_memcpy_1<sizeof(q4), 2>(&q4, kj[ib].qs + qoff + 4*w);
                    dot += qp[4*w + 0]*int8_t(q4 >>  0) + qp[4*w + 1]*int8_t(q4 >>  8) +
                           qp[4*w + 2]*int8_t(q4 >> 16) + qp[4*w + 3]*int8_t(q4 >> 24);
                }
                qk += dot*kd;
            }
            float b_rel = 0.0f;
            const int dist = rel_pos[j];
            if (dist < n_extent) {
                for (int i = li; i < n_rel; i += lanes) {
                    b_rel += table[dist*s_t1 + i]*rs[i];
                }
            }
            part = qk*scale + b_rel;
        }
        part = warp_reduce_sum<lanes>(part);
        if (valid && li == 0) {
            const float l = active ? ((part - sh) + mv) : -INFINITY;
            lg[j] = l;
            mx = fmaxf(mx, l);
        }
    }

    __shared__ float buf[32];
    mx = block_reduce<block_reduce_method::MAX>(mx, buf);
    __syncthreads();

    float sum = 0.0f;
    for (int j = threadIdx.x; j < n_kv; j += blockDim.x) {
        const float lv = lg[j];
        const float e  = lv > -1e30f ? expf(lv - mx) : 0.0f;
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
            const float p = lg[j];
            if (p == 0.0f) {
                continue;
            }
            const block_q8_0 * vj = (const block_q8_0 *) (v + hk*s_v1 + j*s_v2) + ib;
            int q4;
            ggml_cuda_memcpy_1<sizeof(q4), 2>(&q4, vj->qs + 4*iqs);
            const float w = p*__half2float(vj->d);
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

static constexpr int INKLING2_ATTN_SPLIT_TILE = 256;
static constexpr int INKLING2_ATTN_MAX_SPLITS = 256;

// split-KV online-softmax stage 1 for Q8_0 KV cache (unlimited n_kv, mask-skipping)
template <typename M>
static __global__ void inkling2_attn_q8_0_split(
        const float * __restrict__ q, const char * __restrict__ k, const char * __restrict__ v, const float * __restrict__ table,
        const float * __restrict__ r, const float * __restrict__ sinks, const int32_t * __restrict__ rel_pos, const M * __restrict__ mask,
        float * __restrict__ partial, const int d, const int n_gqa, const int n_kv, const int n_rel, const int n_extent,
        const int64_t s_q1, const int64_t s_k1, const int64_t s_k2, const int64_t s_v1, const int64_t s_v2, const int64_t s_t1, const int64_t s_r1,
        const float scale, const int n_rot_q, const int n_splits) {
    constexpr int lanes = 8;

    const int h     = blockIdx.x;
    const int split = blockIdx.y;
    const int hk    = h/n_gqa;

    const int j_beg = (int) (((int64_t) split * n_kv) / n_splits);
    const int j_end = (int) (((int64_t) (split + 1) * n_kv) / n_splits);

    float * out_s = partial + ((size_t) h * n_splits + split) * (d + 2);
    if (j_beg >= j_end) {
        if (threadIdx.x == 0) {
            out_s[d]     = -INFINITY;
            out_s[d + 1] = 0.0f;
        }
        return;
    }

    extern __shared__ float sm[];
    float * lg   = sm;
    float * qs   = lg + INKLING2_ATTN_SPLIT_TILE;
    float * rs   = qs + d;
    float * vbuf = rs + n_rel;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        qs[i] = q[h*s_q1 + i];
    }
    for (int i = threadIdx.x; i < n_rel; i += blockDim.x) {
        rs[i] = r[h*s_r1 + i];
    }
    __syncthreads();
    inkling2_fwht_smem(qs, d, n_rot_q);

    const float sh      = sinks[h];
    const int   li      = threadIdx.x % lanes;
    const int   ng      = blockDim.x / lanes;
    const int   d4      = d / 4;
    const int   nb_half = d / 16;
    const int   ng_v    = blockDim.x / d4;
    const int   vg      = threadIdx.x / d4;
    const int   vc      = threadIdx.x % d4;
    const int   vib     = vc / QI8_0;
    const int   viqs    = vc % QI8_0;

    __shared__ float buf[32];

    float m_run  = -INFINITY;
    float l_run  = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int t0 = j_beg; t0 < j_end; t0 += INKLING2_ATTN_SPLIT_TILE) {
        const int tn = min(INKLING2_ATTN_SPLIT_TILE, j_end - t0);

        float mx = -INFINITY;
        for (int k0 = 0; k0 < tn; k0 += ng) {
            const int   kk     = k0 + threadIdx.x / lanes;
            const bool  valid  = kk < tn;
            const int   j      = t0 + kk;
            const float mv     = valid ? ggml_cuda_cast<float>(mask[j]) : -INFINITY;
            const bool  active = valid && mv > -1e30f;

            float part = 0.0f;
            if (active) {
                const block_q8_0 * kj = (const block_q8_0 *) (k + hk*s_k1 + j*s_k2);
                float qk = 0.0f;
                for (int b = li; b < nb_half; b += lanes) {
                    const int ib   = b >> 1;
                    const int qoff = (b & 1) * 16;
                    const float kd = __half2float(kj[ib].d);
                    const float * qp = qs + b*16;
                    float dot = 0.0f;
#pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        int q4;
                        ggml_cuda_memcpy_1<sizeof(q4), 2>(&q4, kj[ib].qs + qoff + 4*w);
                        dot += qp[4*w + 0]*int8_t(q4 >>  0) + qp[4*w + 1]*int8_t(q4 >>  8) +
                               qp[4*w + 2]*int8_t(q4 >> 16) + qp[4*w + 3]*int8_t(q4 >> 24);
                    }
                    qk += dot*kd;
                }
                float b_rel = 0.0f;
                const int dist = rel_pos[j];
                if (dist < n_extent) {
                    for (int i = li; i < n_rel; i += lanes) {
                        b_rel += table[dist*s_t1 + i]*rs[i];
                    }
                }
                part = qk*scale + b_rel;
            }
            part = warp_reduce_sum<lanes>(part);
            if (valid && li == 0) {
                const float l = active ? ((part - sh) + mv) : -INFINITY;
                lg[kk] = l;
                mx = fmaxf(mx, l);
            }
        }

        const float m_tile = block_reduce<block_reduce_method::MAX>(mx, buf);
        __syncthreads();
        if (m_tile <= -1e30f) {
            continue;
        }

        const float m_new = fmaxf(m_run, m_tile);
        const float alpha = m_run > -1e30f ? expf(m_run - m_new) : 0.0f;

        float sum_tile = 0.0f;
        for (int kk = threadIdx.x; kk < tn; kk += blockDim.x) {
            const float lv = lg[kk];
            const float e  = lv > -1e30f ? expf(lv - m_new) : 0.0f;
            lg[kk]    = e;
            sum_tile += e;
        }
        sum_tile = block_reduce<block_reduce_method::SUM>(sum_tile, buf);
        __syncthreads();

        l_run  = l_run*alpha + sum_tile;
        m_run  = m_new;
        acc[0] *= alpha;
        acc[1] *= alpha;
        acc[2] *= alpha;
        acc[3] *= alpha;

        if (vg < ng_v) {
            for (int kk = vg; kk < tn; kk += ng_v) {
                const float p = lg[kk];
                if (p == 0.0f) {
                    continue;
                }
                const int j = t0 + kk;
                const block_q8_0 * vj = (const block_q8_0 *) (v + hk*s_v1 + j*s_v2) + vib;
                int q4;
                ggml_cuda_memcpy_1<sizeof(q4), 2>(&q4, vj->qs + 4*viqs);
                const float w = p*__half2float(vj->d);
                acc[0] += w*int8_t(q4 >>  0);
                acc[1] += w*int8_t(q4 >>  8);
                acc[2] += w*int8_t(q4 >> 16);
                acc[3] += w*int8_t(q4 >> 24);
            }
        }
        __syncthreads();
    }

    if (l_run == 0.0f) {
        if (threadIdx.x == 0) {
            out_s[d]     = -INFINITY;
            out_s[d + 1] = 0.0f;
        }
        return;
    }

    if (vg < ng_v) {
        float * vp = vbuf + vg*d + 4*vc;
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
        out_s[threadIdx.x] = total;
    }
    if (threadIdx.x == 0) {
        out_s[d]     = m_run;
        out_s[d + 1] = l_run;
    }
}

// reductions over 16 lanes (and 64) with GCN DPP row ops instead of ds_bpermute: xor 1, xor 2 (quad_perm), then
// row_half_mirror (8 lanes) and row_mirror (16 lanes); every lane of the group gets the result
#if defined(GGML_USE_HIP)
template <int ctrl>
static __device__ __forceinline__ float inkling2_dpp(const float x) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), ctrl, 0xF, 0xF, false));
}
#endif

static __device__ __forceinline__ float inkling2_sum16(float x) {
#if defined(GGML_USE_HIP)
    x += inkling2_dpp<0xB1>(x);
    x += inkling2_dpp<0x4E>(x);
    x += inkling2_dpp<0x141>(x);
    x += inkling2_dpp<0x140>(x);
    return x;
#else
    return warp_reduce_sum<16>(x);
#endif
}

static __device__ __forceinline__ float inkling2_max16(float x) {
#if defined(GGML_USE_HIP)
    x = fmaxf(x, inkling2_dpp<0xB1>(x));
    x = fmaxf(x, inkling2_dpp<0x4E>(x));
    x = fmaxf(x, inkling2_dpp<0x141>(x));
    x = fmaxf(x, inkling2_dpp<0x140>(x));
    return x;
#else
    return warp_reduce_max<16>(x);
#endif
}

// over a 64-lane wave: 16 lanes with DPP, then two shuffles
static __device__ __forceinline__ float inkling2_sum64(float x) {
    x = inkling2_sum16(x);
    x += __shfl_xor(x, 16, 64);
    x += __shfl_xor(x, 32, 64);
    return x;
}

static __device__ __forceinline__ float inkling2_sum32(float x) {
    x = inkling2_sum16(x);
    x += __shfl_xor(x, 16, 64);
    return x;
}

static __device__ __forceinline__ float inkling2_max32(float x) {
    x = inkling2_max16(x);
    x = fmaxf(x, __shfl_xor(x, 16, 64));
    return x;
}

static __device__ __forceinline__ float inkling2_max64(float x) {
    x = inkling2_max16(x);
    x = fmaxf(x, __shfl_xor(x, 16, 64));
    x = fmaxf(x, __shfl_xor(x, 32, 64));
    return x;
}

// wave-level LDS ordering: the lanes of one wave exchange values through LDS without a block barrier
static __device__ __forceinline__ void inkling2_wave_sync() {
#if defined(GGML_USE_HIP)
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#else
    __syncwarp();
#endif
}

static __device__ __forceinline__ bool inkling2_wave_all(const bool pred) {
#if defined(GGML_USE_HIP)
    return __all(pred);
#else
    return __all_sync(0xffffffff, pred);
#endif
}

static __device__ __forceinline__ bool inkling2_wave_any(const bool pred) {
#if defined(GGML_USE_HIP)
    return __any(pred);
#else
    return __any_sync(0xffffffff, pred);
#endif
}

static __device__ __forceinline__ uint4 inkling2_load_q8_16(const void * p) {
#if defined(GGML_USE_HIP)
    return *((const uint4 *) p);
#else
    uint4 out;
    ggml_cuda_memcpy_1<sizeof(out), 2>(&out, p);
    return out;
#endif
}

static __device__ __forceinline__ int inkling2_load_q8_4(const void * p) {
#if defined(GGML_USE_HIP)
    return *((const int *) p);
#else
    int out;
    ggml_cuda_memcpy_1<sizeof(out), 2>(&out, p);
    return out;
#endif
}

// split-KV decode attention for Q8_0 KV cache (d = 128, 64-lane waves):
// each block packs up to 4 KV heads of the same split on one CU so their waves share L1 for mask, rel_pos, and KV lines.
// K uses 8 lanes/cell (16 dims, uint4 loads, w-outer qs_k reuse); V uses 32 lanes/cell (4 dims, <= 84 VGPRs, 0 spills).
static constexpr int INKLING2_ATTN_WAVES = 4;


template <typename M, int G>
static __global__ void __launch_bounds__(64*INKLING2_ATTN_WAVES, 3) inkling2_attn_q8_0_split_gqa(
        const float * __restrict__ q, const char * __restrict__ k, const char * __restrict__ v, const float * __restrict__ table,
        const float * __restrict__ r, const float * __restrict__ sinks, const int32_t * __restrict__ rel_pos, const M * __restrict__ mask,
        float * __restrict__ partial, const int n_head_kv, const int n_kv, const int n_rel, const int n_extent,
        const int64_t s_q1, const int64_t s_k1, const int64_t s_k2, const int64_t s_v1, const int64_t s_v2, const int64_t s_t1, const int64_t s_r1,
        const float scale, const int n_rot_q, const int n_splits) {
    constexpr int D     = 128;
    constexpr int CHUNK = 64;

#if defined(GGML_USE_HIP)
    // tell LLVM wave index is uniform across all 64 lanes so hk, kh, vh, sh, and LDS slices stay in SGPRs
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
#else
    const int wave = threadIdx.x / 64;
#endif
    const int lane = threadIdx.x & 63;
    const int hk   = blockIdx.x*INKLING2_ATTN_WAVES + wave;
    const int sp   = blockIdx.y;
    if (hk >= n_head_kv || sp >= n_splits) {
        return;
    }

    __shared__ __align__(16) float  qs_tmp[INKLING2_ATTN_WAVES][D];
    __shared__ __align__(16) float4 qs_k[INKLING2_ATTN_WAVES][G][4][8]; // transposed by 16B word (0 LDS bank conflicts)
    __shared__               float  rs[INKLING2_ATTN_WAVES][G][16];
    __shared__               float  lgs[INKLING2_ATTN_WAVES][G][CHUNK];
    __shared__               float  mvs[INKLING2_ATTN_WAVES][CHUNK];
    __shared__               int    dss[INKLING2_ATTN_WAVES][CHUNK];

    // per-wave Q FWHT + transpose into qs_k without block barriers
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float * qg = q + (int64_t) (hk*G + g)*s_q1;
        qs_tmp[wave][lane]      = qg[lane];
        qs_tmp[wave][lane + 64] = qg[lane + 64];
        inkling2_wave_sync();
        if (n_rot_q == 128) {
            const float inv = rsqrtf(128.0f);
            float a = qs_tmp[wave][lane]*inv;
            float b = qs_tmp[wave][lane + 64]*inv;
#pragma unroll
            for (int h = 1; h < 64; h *= 2) {
                const float a2 = __shfl_xor(a, h, 64);
                const float b2 = __shfl_xor(b, h, 64);
                a = (lane & h) == 0 ? a + a2 : a2 - a;
                b = (lane & h) == 0 ? b + b2 : b2 - b;
            }
            qs_tmp[wave][lane]      = a + b;
            qs_tmp[wave][lane + 64] = a - b;
            inkling2_wave_sync();
        } else if (n_rot_q > 0) {
            for (int h = 1; h < n_rot_q; h *= 2) {
                const float a0 = qs_tmp[wave][lane];
                const float b0 = qs_tmp[wave][lane ^ h];
                const float a1 = qs_tmp[wave][lane + 64];
                const float b1 = qs_tmp[wave][(lane + 64) ^ h];
                inkling2_wave_sync();
                qs_tmp[wave][lane]      = (lane & h) == 0 ? a0 + b0 : b0 - a0;
                qs_tmp[wave][lane + 64] = ((lane + 64) & h) == 0 ? a1 + b1 : b1 - a1;
                inkling2_wave_sync();
            }
            const float inv = rsqrtf((float) n_rot_q);
            qs_tmp[wave][lane]      *= inv;
            qs_tmp[wave][lane + 64] *= inv;
            inkling2_wave_sync();
        }
        if (lane < D/4) {
            const int l8i = lane >> 2;
            const int wi  = lane & 3;
            const float4 v = ((const float4 *) qs_tmp[wave])[lane];
            qs_k[wave][g][wi][l8i] = make_float4(v.x*scale, v.y*scale, v.z*scale, v.w*scale);
        }
        if (lane < 16) {
            rs[wave][g][lane] = lane < n_rel ? r[(int64_t) (hk*G + g)*s_r1 + lane] : 0.0f;
        }
    }
    inkling2_wave_sync();

    const int j_beg = (int) (((int64_t) sp * n_kv) / n_splits);
    const int j_end = (int) (((int64_t) (sp + 1) * n_kv) / n_splits);

    float sh[G];
    float m_run[G];
    float l_run[G];
    float acc[G][4]; // dims 4*l32 .. 4*l32 + 3 of cells sub32, sub32 + 2, ...
#pragma unroll
    for (int g = 0; g < G; ++g) {
        sh[g]    = sinks[hk*G + g];
        m_run[g] = -INFINITY;
        l_run[g] = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[g][i] = 0.0f;
        }
    }

    const char *  kh      = k + hk*s_k1;
    const char *  vh      = v + hk*s_v1;
    const int64_t step8_k = 8*s_k2;
    const int64_t step2_v = 2*s_v2;

    for (int c0 = j_beg; c0 < j_end; c0 += CHUNK) {
        const int cn = min(CHUNK, j_end - c0);

        const bool  valid_l = lane < cn;
        const float mv_l    = valid_l ? ggml_cuda_cast<float>(mask[c0 + lane]) : -INFINITY;
        const int   ds_l    = valid_l ? rel_pos[c0 + lane] : n_extent;
        const bool  fast_k  = (cn == CHUNK) && inkling2_wave_all(mv_l == 0.0f) && !inkling2_wave_any(ds_l < n_extent);

        {
            const int l8    = lane & 7;
            const int sub8  = lane >> 3;
            const int kib8  = l8 >> 1;
            const int kofs8 = 16*(l8 & 1);

            if (fast_k) {
                const char * kp = kh + (int64_t) (c0 + sub8)*s_k2 + (int64_t) kib8 * sizeof(block_q8_0);
#pragma unroll 1
                for (int b = 0; b < CHUNK/16; ++b) {
                    uint4 q16[2];
                    half  kd_h[2];
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        const block_q8_0 * kb = (const block_q8_0 *) (kp + u*step8_k);
                        q16[u]  = inkling2_load_q8_16(kb->qs + kofs8);
                        kd_h[u] = kb->d;
                    }
                    kp += 2*step8_k;

                    int l8_v = l8;
#if defined(GGML_USE_HIP)
                    asm("" : "+v"(l8_v));
#endif
                    const uint32_t qw0[4] = { q16[0].x, q16[0].y, q16[0].z, q16[0].w };
                    const uint32_t qw1[4] = { q16[1].x, q16[1].y, q16[1].z, q16[1].w };
                    float dot[2][G];
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        dot[0][g] = 0.0f;
                        dot[1][g] = 0.0f;
                    }
#pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        const float a0 = int8_t(qw0[w] >>  0);
                        const float a1 = int8_t(qw0[w] >>  8);
                        const float a2 = int8_t(qw0[w] >> 16);
                        const float a3 = int8_t(qw0[w] >> 24);
                        const float b0 = int8_t(qw1[w] >>  0);
                        const float b1 = int8_t(qw1[w] >>  8);
                        const float b2 = int8_t(qw1[w] >> 16);
                        const float b3 = int8_t(qw1[w] >> 24);
#pragma unroll
                        for (int g = 0; g < G; ++g) {
                            const float4 qv = qs_k[wave][g][w][l8_v];
                            dot[0][g] += qv.x*a0 + qv.y*a1 + qv.z*a2 + qv.w*a3;
                            dot[1][g] += qv.x*b0 + qv.y*b1 + qv.z*b2 + qv.w*b3;
                        }
                    }
                    const float kd0 = __half2float(kd_h[0]);
                    const float kd1 = __half2float(kd_h[1]);
                    float part[2][G];
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        part[0][g] = dot[0][g]*kd0;
                        part[1][g] = dot[1][g]*kd1;
                    }
#if defined(GGML_USE_HIP)
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
#pragma unroll
                        for (int g = 0; g < G; ++g) { part[u][g] += inkling2_dpp<0xB1>(part[u][g]); }
                    }
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
#pragma unroll
                        for (int g = 0; g < G; ++g) { part[u][g] += inkling2_dpp<0x4E>(part[u][g]); }
                    }
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
#pragma unroll
                        for (int g = 0; g < G; ++g) { part[u][g] += inkling2_dpp<0x141>(part[u][g]); }
                    }
#else
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
#pragma unroll
                        for (int g = 0; g < G; ++g) { part[u][g] = warp_reduce_sum<8>(part[u][g]); }
                    }
#endif
                    if (l8 == 0) {
                        const int cc0 = 16*b + sub8;
#pragma unroll
                        for (int g = 0; g < G; ++g) {
                            lgs[wave][g][cc0 + 0] = part[0][g] - sh[g];
                            lgs[wave][g][cc0 + 8] = part[1][g] - sh[g];
                        }
                    }
                }
            } else {
                mvs[wave][lane] = mv_l;
                dss[wave][lane] = ds_l;
                inkling2_wave_sync();

#pragma unroll 1
                for (int st = 0; st < CHUNK/8; ++st) {
                    const int   cc     = 8*st + sub8;
                    const bool  valid  = cc < cn;
                    const int   j      = c0 + (valid ? cc : 0);
                    const float mv     = mvs[wave][cc];
                    const bool  active = mv > -1e30f;

                    const block_q8_0 * kb = (const block_q8_0 *) (kh + (int64_t) j*s_k2) + kib8;
                    const uint4 q16  = inkling2_load_q8_16(kb->qs + kofs8);
                    const float kd   = __half2float(kb->d);
                    const int   dist = dss[wave][cc];
                    const float tb0  = (dist < n_extent && 2*l8 + 0 < n_rel) ? table[dist*s_t1 + 2*l8 + 0] : 0.0f;
                    const float tb1  = (dist < n_extent && 2*l8 + 1 < n_rel) ? table[dist*s_t1 + 2*l8 + 1] : 0.0f;

                    int l8_v = l8;
#if defined(GGML_USE_HIP)
                    asm("" : "+v"(l8_v));
#endif
                    const uint32_t qw[4] = { q16.x, q16.y, q16.z, q16.w };
                    float dot[G];
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        dot[g] = 0.0f;
                    }
#pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        const float f0 = int8_t(qw[w] >>  0);
                        const float f1 = int8_t(qw[w] >>  8);
                        const float f2 = int8_t(qw[w] >> 16);
                        const float f3 = int8_t(qw[w] >> 24);
#pragma unroll
                        for (int g = 0; g < G; ++g) {
                            const float4 qv = qs_k[wave][g][w][l8_v];
                            dot[g] += qv.x*f0 + qv.y*f1 + qv.z*f2 + qv.w*f3;
                        }
                    }
                    float part[G];
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        part[g] = dot[g]*kd + tb0*rs[wave][g][2*l8 + 0] + tb1*rs[wave][g][2*l8 + 1];
                    }
#if defined(GGML_USE_HIP)
#pragma unroll
                    for (int g = 0; g < G; ++g) { part[g] += inkling2_dpp<0xB1>(part[g]); }
#pragma unroll
                    for (int g = 0; g < G; ++g) { part[g] += inkling2_dpp<0x4E>(part[g]); }
#pragma unroll
                    for (int g = 0; g < G; ++g) { part[g] += inkling2_dpp<0x141>(part[g]); }
#else
#pragma unroll
                    for (int g = 0; g < G; ++g) { part[g] = warp_reduce_sum<8>(part[g]); }
#endif
                    if (l8 == 0) {
#pragma unroll
                        for (int g = 0; g < G; ++g) {
                            lgs[wave][g][cc] = valid && active ? ((part[g] - sh[g]) + mv) : -INFINITY;
                        }
                    }
                }
            }
        }
        inkling2_wave_sync();

        // online softmax over the chunk, one cell per lane
        float alpha[G];
        bool  any = false;
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float l     = lgs[wave][g][lane];
            const float m_c   = inkling2_max64(l);
            const float m_new = fmaxf(m_run[g], m_c);
            alpha[g]          = m_run[g] > -1e30f ? expf(m_run[g] - m_new) : 0.0f;
            const float p     = l > -1e30f ? expf(l - m_new) : 0.0f;
            const float sum   = inkling2_sum64(p);
            lgs[wave][g][lane] = p;
            if (m_c > -1e30f) {
                any      = true;
                l_run[g] = l_run[g]*alpha[g] + sum;
                m_run[g] = m_new;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    acc[g][i] *= alpha[g];
                }
            }
        }
        inkling2_wave_sync();
        if (!any) {
            continue;
        }

        // V: 32 lanes per cell (4 dims each), batched 4 steps at a time on full chunks
        {
            const int l32    = lane & 31;
            const int sub32  = lane >> 5;
            const int vib32  = l32 >> 3;
            const int vofs32 = 4*(l32 & 7);

            if (cn == CHUNK) {
                const char * vp = vh + (int64_t) (c0 + sub32)*s_v2 + (int64_t) vib32 * sizeof(block_q8_0);
#pragma unroll 1
                for (int b = 0; b < CHUNK/8; ++b) {
                    int  q4[4];
                    half vd_h[4];
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const block_q8_0 * vb = (const block_q8_0 *) (vp + u*step2_v);
                        q4[u]   = inkling2_load_q8_4(vb->qs + vofs32);
                        vd_h[u] = vb->d;
                    }
                    vp += 4*step2_v;
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const float f0 = int8_t(q4[u] >>  0);
                        const float f1 = int8_t(q4[u] >>  8);
                        const float f2 = int8_t(q4[u] >> 16);
                        const float f3 = int8_t(q4[u] >> 24);
                        const float vd = __half2float(vd_h[u]);
                        const int   cc = 8*b + 2*u + sub32;
#pragma unroll
                        for (int g = 0; g < G; ++g) {
                            const float pvd = lgs[wave][g][cc]*vd;
                            acc[g][0] += pvd*f0;
                            acc[g][1] += pvd*f1;
                            acc[g][2] += pvd*f2;
                            acc[g][3] += pvd*f3;
                        }
                    }
                }
            } else {
#pragma unroll 1
                for (int st = 0; st < CHUNK/2; ++st) {
                    const int cc = 2*st + sub32;
                    if (cc < cn) {
                        const block_q8_0 * vb = (const block_q8_0 *) (vh + (int64_t) (c0 + cc)*s_v2) + vib32;
                        const int   q4 = inkling2_load_q8_4(vb->qs + vofs32);
                        const float vd = __half2float(vb->d);
                        const float f0 = int8_t(q4 >>  0);
                        const float f1 = int8_t(q4 >>  8);
                        const float f2 = int8_t(q4 >> 16);
                        const float f3 = int8_t(q4 >> 24);
#pragma unroll
                        for (int g = 0; g < G; ++g) {
                            const float pvd = lgs[wave][g][cc]*vd;
                            acc[g][0] += pvd*f0;
                            acc[g][1] += pvd*f1;
                            acc[g][2] += pvd*f2;
                            acc[g][3] += pvd*f3;
                        }
                    }
                }
            }
        }
        inkling2_wave_sync();
    }

    const int l32   = lane & 31;
    const int sub32 = lane >> 5;
#pragma unroll
    for (int g = 0; g < G; ++g) {
        float * out_s = partial + ((size_t) (hk*G + g) * n_splits + sp) * (D + 2);
        if (l_run[g] == 0.0f) {
            if (lane == 0) {
                out_s[D]     = -INFINITY;
                out_s[D + 1] = 0.0f;
            }
            continue;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float a = acc[g][i];
            a += __shfl_xor(a, 32, 64);
            if (sub32 == 0) {
                out_s[4*l32 + i] = a;
            }
        }
        if (lane == 0) {
            out_s[D]     = m_run[g];
            out_s[D + 1] = l_run[g];
        }
    }
}

// stage 2 combine: merge split-KV partial sums with the zero sink, apply V FWHT and output gate
static __global__ void inkling2_attn_combine(
        const float * __restrict__ partial, const float * __restrict__ gate, float * __restrict__ dst,
        const int d, const int n_splits, const int n_rot_v) {
    extern __shared__ float qs[];
    __shared__ float sm_m[INKLING2_ATTN_MAX_SPLITS];
    __shared__ float sm_l[INKLING2_ATTN_MAX_SPLITS];
    __shared__ float sm_w[INKLING2_ATTN_MAX_SPLITS];
    __shared__ float sm_l_global;

    constexpr int pws = ggml_cuda_get_physical_warp_size();
    const int h = blockIdx.x;
    const int i = threadIdx.x;
    const float * p_head = partial + (size_t) h * n_splits * (d + 2);

    // load all split (m, l) pairs in parallel and compute normalized weights in warp 0
    for (int s = i; s < n_splits; s += blockDim.x) {
        const float * ps = p_head + (size_t) s * (d + 2);
        sm_m[s] = ps[d];
        sm_l[s] = ps[d + 1];
    }
    __syncthreads();

    if (i < pws) {
        float mv = 0.0f; // zero sink
        for (int s = i; s < n_splits; s += pws) {
            if (sm_l[s] > 0.0f) {
                mv = fmaxf(mv, sm_m[s]);
            }
        }
        const float m_global = warp_reduce_max<pws>(mv);
        float lv = 0.0f;
        for (int s = i; s < n_splits; s += pws) {
            const float ls = sm_l[s];
            const float w  = ls > 0.0f ? expf(sm_m[s] - m_global) : 0.0f;
            sm_w[s] = w;
            lv     += ls * w;
        }
        const float l_global = warp_reduce_sum<pws>(lv) + expf(0.0f - m_global);
        if (i == 0) {
            sm_l_global = l_global;
        }
    }
    __syncthreads();

    const int ng = max(1, (int) blockDim.x / d);
    const int g  = i / d;
    const int di = i % d;
    if (g < ng) {
        float o = 0.0f;
        for (int s = g; s < n_splits; s += ng) {
            const float w = sm_w[s];
            if (w > 0.0f) {
                o += p_head[(size_t) s * (d + 2) + di] * w;
            }
        }
        qs[g*d + di] = o;
    }
    __syncthreads();

    if (i < d) {
        float total = qs[i];
        for (int gg = 1; gg < ng; ++gg) {
            total += qs[gg*d + i];
        }
        qs[i] = total / sm_l_global;
    }
    __syncthreads();

    inkling2_fwht_smem(qs, d, n_rot_v);
    if (i < d) {
        const float gt = gate[h*d + i];
        dst[h*d + i] = qs[i]*(1.0f/(1.0f + expf(-gt)));
    }
}

// split-KV online-softmax stage 1 for F32/F16/BF16 KV cache (unlimited n_kv, mask-skipping)
template <typename T, typename M, bool v_trans, bool vec>
static __global__ void inkling2_attn_split_f32(
        const float * __restrict__ q, const T * __restrict__ k, const T * __restrict__ v, const float * __restrict__ table,
        const float * __restrict__ r, const float * __restrict__ sinks, const int32_t * __restrict__ rel_pos, const M * __restrict__ mask,
        float * __restrict__ partial, const int d, const int n_gqa, const int n_kv, const int n_rel, const int n_extent,
        const int64_t s_q1, const int64_t s_k1, const int64_t s_k2, const int64_t s_v1, const int64_t s_v2, const int64_t s_t1, const int64_t s_r1,
        const float scale, const int n_rot_q, const int n_splits) {
    constexpr int vk    = vec ? 16/sizeof(T) : 1;
    constexpr int lanes = 16;

    const int h     = blockIdx.x;
    const int split = blockIdx.y;
    const int hk    = h/n_gqa;

    const int j_beg = (int) (((int64_t) split * n_kv) / n_splits);
    const int j_end = (int) (((int64_t) (split + 1) * n_kv) / n_splits);

    float * out_s = partial + ((size_t) h * n_splits + split) * (d + 2);
    if (j_beg >= j_end) {
        if (threadIdx.x == 0) {
            out_s[d]     = -INFINITY;
            out_s[d + 1] = 0.0f;
        }
        return;
    }

    extern __shared__ float sm[];
    float * lg   = sm;
    float * qs   = lg + INKLING2_ATTN_SPLIT_TILE;
    float * rs   = qs + d;
    float * vbuf = rs + n_rel;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        qs[i] = q[h*s_q1 + i];
    }
    for (int i = threadIdx.x; i < n_rel; i += blockDim.x) {
        rs[i] = r[h*s_r1 + i];
    }
    if constexpr (v_trans) {
        for (int i = threadIdx.x; i < d; i += blockDim.x) {
            vbuf[i] = 0.0f;
        }
    }
    __syncthreads();
    inkling2_fwht_smem(qs, d, n_rot_q);

    const float sh   = sinks[h];
    const int   li   = threadIdx.x % lanes;
    const int   ng   = blockDim.x / lanes;
    const int   warp = threadIdx.x / WARP_SIZE;
    const int   lane = threadIdx.x % WARP_SIZE;
    const int   nw   = blockDim.x / WARP_SIZE;
    const int   ng_v = blockDim.x / d;
    const int   vg   = threadIdx.x / d;
    const int   vi_i = threadIdx.x % d;

    __shared__ float buf[32];

    float m_run = -INFINITY;
    float l_run = 0.0f;
    float acc_v = 0.0f;

    for (int t0 = j_beg; t0 < j_end; t0 += INKLING2_ATTN_SPLIT_TILE) {
        const int tn = min(INKLING2_ATTN_SPLIT_TILE, j_end - t0);

        float mx = -INFINITY;
        for (int k0 = 0; k0 < tn; k0 += ng) {
            const int   kk     = k0 + threadIdx.x / lanes;
            const bool  valid  = kk < tn;
            const int   j      = t0 + kk;
            const float mv     = valid ? ggml_cuda_cast<float>(mask[j]) : -INFINITY;
            const bool  active = valid && mv > -1e30f;

            float part = 0.0f;
            if (active) {
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
                const float l = active ? ((part - sh) + mv) : -INFINITY;
                lg[kk] = l;
                mx = fmaxf(mx, l);
            }
        }

        const float m_tile = block_reduce<block_reduce_method::MAX>(mx, buf);
        __syncthreads();
        if (m_tile <= -1e30f) {
            continue;
        }

        const float m_new = fmaxf(m_run, m_tile);
        const float alpha = m_run > -1e30f ? expf(m_run - m_new) : 0.0f;

        float sum_tile = 0.0f;
        for (int kk = threadIdx.x; kk < tn; kk += blockDim.x) {
            const float lv = lg[kk];
            const float e  = lv > -1e30f ? expf(lv - m_new) : 0.0f;
            lg[kk]    = e;
            sum_tile += e;
        }
        sum_tile = block_reduce<block_reduce_method::SUM>(sum_tile, buf);
        __syncthreads();

        l_run = l_run*alpha + sum_tile;
        m_run = m_new;

        if constexpr (v_trans) {
            if (alpha != 1.0f) {
                for (int i = threadIdx.x; i < d; i += blockDim.x) {
                    vbuf[i] *= alpha;
                }
            }
            __syncthreads();
            for (int i = warp; i < d; i += nw) {
                float acc = 0.0f;
                const T * vi = v + hk*s_v1 + i*s_v2 + t0;
                for (int kk = lane; kk < tn; kk += WARP_SIZE) {
                    const float p = lg[kk];
                    if (p != 0.0f) {
                        acc += p*ggml_cuda_cast<float>(vi[kk]);
                    }
                }
                acc = warp_reduce_sum(acc);
                if (lane == 0) {
                    vbuf[i] += acc;
                }
            }
            __syncthreads();
        } else {
            acc_v *= alpha;
            if (vg < ng_v) {
                for (int kk = vg; kk < tn; kk += ng_v) {
                    const float p = lg[kk];
                    if (p != 0.0f) {
                        acc_v += p*ggml_cuda_cast<float>(v[vi_i + hk*s_v1 + (t0 + kk)*s_v2]);
                    }
                }
            }
            __syncthreads();
        }
    }

    if (l_run == 0.0f) {
        if (threadIdx.x == 0) {
            out_s[d]     = -INFINITY;
            out_s[d + 1] = 0.0f;
        }
        return;
    }

    if constexpr (v_trans) {
        if (threadIdx.x < d) {
            out_s[threadIdx.x] = vbuf[threadIdx.x];
        }
    } else {
        if (vg < ng_v) {
            vbuf[vg*d + vi_i] = acc_v;
        }
        __syncthreads();
        if (threadIdx.x < d) {
            float total = 0.0f;
            for (int g = 0; g < ng_v; ++g) {
                total += vbuf[g*d + threadIdx.x];
            }
            out_s[threadIdx.x] = total;
        }
    }
    if (threadIdx.x == 0) {
        out_s[d]     = m_run;
        out_s[d + 1] = l_run;
    }
}

// tiled causal prefill attention with 1D cell/query positions (full or standard SWA), D = 128, BQ = 32, BK = 64
template <bool kv_q8_0>
static __global__ void __launch_bounds__(256, 2) inkling2_attn_prefill_tile(
        const float * __restrict__ q, const char * __restrict__ k, const char * __restrict__ v,
        const float * __restrict__ table, const float * __restrict__ r, const float * __restrict__ sinks,
        const int32_t * __restrict__ cell_pos, const int32_t * __restrict__ q_pos,
        const float * __restrict__ gate, float * __restrict__ dst,
        const int T, const int n_head, const int n_gqa, const int n_kv, const int n_rel, const int n_extent, const int n_swa,
        const int64_t s_q1, const int64_t s_q2, const int64_t s_k1, const int64_t s_k2,
        const int64_t s_v1, const int64_t s_v2, const int64_t s_t1, const int64_t s_r1, const int64_t s_r2,
        const int64_t s_s1, const int64_t s_g1, const int64_t s_dst1, const float scale) {
    constexpr int D  = 128;
    constexpr int BQ = 32;
    constexpr int BK = 64;

    const int q0   = blockIdx.x * BQ;
    const int qn   = min(BQ, T - q0);
    const int h    = blockIdx.y;
    const int hk   = h / n_gqa;
    const int tid  = threadIdx.x;
    const int warp = tid / 64;
    const int lane = tid % 64;

    // 30.5 KiB LDS total -> 2 blocks (8 waves) per CU; s_KV is XOR-swizzled across 16-byte chunks (0 bank conflicts)
    __shared__ __align__(16) half2   s_Q[BQ][D/2];
    __shared__ __align__(16) half2   s_KV[BK][D/2];
    __shared__ __align__(16) half2   s_P[BQ][BK/2];
    __shared__ __align__(16) float   s_R[BQ][16];
    __shared__               float   s_sinks[BQ];
    __shared__               int32_t s_qpos[BQ];
    __shared__               int32_t s_cpos[BK];

    for (int idx = tid; idx < BQ*(D/2); idx += 256) {
        const int qi = idx / (D/2);
        const int j  = idx % (D/2);
        if (qi < qn) {
            const float * qp = q + int64_t(q0 + qi)*s_q2 + int64_t(h)*s_q1;
            s_Q[qi][j] = make_half2(qp[2*j]*scale, qp[2*j + 1]*scale);
        } else {
            s_Q[qi][j] = make_half2(0.0f, 0.0f);
        }
    }
    for (int idx = tid; idx < BQ*16; idx += 256) {
        const int qi = idx / 16;
        const int rr = idx % 16;
        s_R[qi][rr] = (qi < qn && rr < n_rel) ? r[int64_t(q0 + qi)*s_r2 + int64_t(h)*s_r1 + rr] : 0.0f;
    }
    if (tid < BQ) {
        s_sinks[tid] = tid < qn ? sinks[int64_t(q0 + tid)*s_s1 + h] : 0.0f;
        s_qpos[tid]  = tid < qn ? q_pos[q0 + tid] : -1;
    }
    __syncthreads();

    int32_t p_max = -1;
    int32_t p_min = INT32_MAX;
    for (int qi = 0; qi < qn; ++qi) {
        p_max = max(p_max, s_qpos[qi]);
        p_min = min(p_min, s_qpos[qi]);
    }
    const int32_t p_lo = n_swa > 0 ? p_min - n_swa + 1 : 0;

    const int n_active = min(n_kv, max(0, cell_pos[n_kv]));
    const char * k_head = k + int64_t(hk)*s_k1;
    const char * v_head = v + int64_t(hk)*s_v1;

    float  m_run[8];
    float  l_run[8];
    float2 o_acc[8];
#pragma unroll
    for (int qi = 0; qi < 8; ++qi) {
        m_run[qi] = -INFINITY;
        l_run[qi] = 0.0f;
        o_acc[qi] = make_float2(0.0f, 0.0f);
    }

    auto load_kv_full = [&] __device__ (const char * base, const int64_t stride_b, const int k0) {
        if constexpr (kv_q8_0) {
            for (int step = 0; step < 8; ++step) {
                const int idx = tid + step*256;
                const int ki  = idx / 32;
                const int c   = idx % 32;
                half2 h0 = make_half2(0.0f, 0.0f);
                half2 h1 = make_half2(0.0f, 0.0f);
                if (k0 + ki < n_active && s_cpos[ki] >= 0) {
                    const block_q8_0 * blk = (const block_q8_0 *) (base + int64_t(k0 + ki)*stride_b) + (c/8);
                    int q4;
                    ggml_cuda_memcpy_1<sizeof(q4), 2>(&q4, blk->qs + 4*(c % 8));
                    const float d = __half2float(blk->d);
                    h0 = make_half2(d*int8_t(q4 >>  0), d*int8_t(q4 >>  8));
                    h1 = make_half2(d*int8_t(q4 >> 16), d*int8_t(q4 >> 24));
                }
                const int sc4 = (c >> 1) ^ (ki & 15);
                const int sub = c & 1;
                s_KV[ki][4*sc4 + 2*sub + 0] = h0;
                s_KV[ki][4*sc4 + 2*sub + 1] = h1;
            }
        } else {
            for (int step = 0; step < 4; ++step) {
                const int idx = tid + step*256;
                const int ki  = idx / 16;
                const int c4  = idx % 16;
                uint4 zv = make_uint4(0, 0, 0, 0);
                if (k0 + ki < n_active && s_cpos[ki] >= 0) {
                    zv = ((const uint4 *) (base + int64_t(k0 + ki)*stride_b))[c4];
                }
                ((uint4 *) s_KV[ki])[c4 ^ (ki & 15)] = zv;
            }
        }
    };

    half * s_P_half = (half *) &s_P[0][0];
    const int lane_c4  = lane >> 2;
    const int lane_off = lane & 3;

    for (int k0 = 0; k0 < n_active; k0 += BK) {
        int32_t p0_load = -1;
        if (tid < BK) {
            p0_load = (k0 + tid < n_active) ? cell_pos[k0 + tid] : -1;
            s_cpos[tid] = p0_load;
        }
        if (!__syncthreads_or(tid < BK && p0_load >= 0 && p0_load <= p_max && p0_load >= p_lo)) {
            continue;
        }

        load_kv_full(k_head, s_k2, k0);
        __syncthreads();

        float s[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int c4 = 0; c4 < 16; ++c4) {
            const int sc4 = c4 ^ (lane & 15);
            __align__(16) half2 kv[4];
            ggml_cuda_memcpy_1<16>(kv, &s_KV[lane][4*sc4]);
#pragma unroll
            for (int qi = 0; qi < 8; ++qi) {
                __align__(16) half2 qv[4];
                ggml_cuda_memcpy_1<16>(qv, &s_Q[warp*8 + qi][4*c4]);
                ggml_cuda_mad(s[qi], qv[0], kv[0]);
                ggml_cuda_mad(s[qi], qv[1], kv[1]);
                ggml_cuda_mad(s[qi], qv[2], kv[2]);
                ggml_cuda_mad(s[qi], qv[3], kv[3]);
            }
        }

        const int32_t p0 = s_cpos[lane];
#pragma unroll
        for (int qi = 0; qi < 8; ++qi) {
            const int q_local = warp*8 + qi;
            if (q_local >= qn) {
                s_P_half[q_local*BK + lane] = __float2half(0.0f);
                continue;
            }
            const int32_t qp = s_qpos[q_local];
            float val = -INFINITY;
            if (k0 + lane < n_active && p0 >= 0 && p0 <= qp && (n_swa == 0 || qp - p0 < n_swa)) {
                const int32_t dist = qp - p0;
                float bias = -s_sinks[q_local];
                if (dist < n_extent) {
                    const float * tb = table + int64_t(dist)*s_t1;
                    if (n_rel == 16 && (s_t1 & 3) == 0) {
                        const float4 * tb4 = (const float4 *) tb;
                        const float4 * rr4 = (const float4 *) s_R[q_local];
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            const float4 a = tb4[k];
                            const float4 b = rr4[k];
                            bias += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
                        }
                    } else {
                        for (int rr = 0; rr < n_rel; ++rr) {
                            bias += tb[rr]*s_R[q_local][rr];
                        }
                    }
                }
                val = s[qi] + bias;
            }
            const float m_tile = inkling2_max64(val);
            float p = 0.0f;
            if (m_tile > -INFINITY) {
                const float m_new = fmaxf(m_run[qi], m_tile);
                const float alpha = m_run[qi] > -INFINITY ? expf(m_run[qi] - m_new) : 0.0f;
                p = val > -INFINITY ? expf(val - m_new) : 0.0f;
                const float l_tile = inkling2_sum64(p);
                l_run[qi] = l_run[qi]*alpha + l_tile;
                m_run[qi] = m_new;
                o_acc[qi].x *= alpha;
                o_acc[qi].y *= alpha;
            }
            s_P_half[q_local*BK + lane] = __float2half(p);
        }

        __syncthreads();
        load_kv_full(v_head, s_v2, k0);
        __syncthreads();

        // lane owns dim pair (2*lane, 2*lane + 1); v_dot2_f32_f16 accumulates 2 keys per instruction into FP32
        for (int kk = 0; kk < BK; kk += 4) {
            const half2 v0 = s_KV[kk + 0][4*(lane_c4 ^ ((kk + 0) & 15)) + lane_off];
            const half2 v1 = s_KV[kk + 1][4*(lane_c4 ^ ((kk + 1) & 15)) + lane_off];
            const half2 v2 = s_KV[kk + 2][4*(lane_c4 ^ ((kk + 2) & 15)) + lane_off];
            const half2 v3 = s_KV[kk + 3][4*(lane_c4 ^ ((kk + 3) & 15)) + lane_off];
            const half2 vlo01 = __halves2half2(__low2half(v0),  __low2half(v1));
            const half2 vhi01 = __halves2half2(__high2half(v0), __high2half(v1));
            const half2 vlo23 = __halves2half2(__low2half(v2),  __low2half(v3));
            const half2 vhi23 = __halves2half2(__high2half(v2), __high2half(v3));
#pragma unroll
            for (int qi = 0; qi < 8; ++qi) {
                __align__(8) half2 pv[2];
                ggml_cuda_memcpy_1<8>(pv, &s_P[warp*8 + qi][kk/2]);
                ggml_cuda_mad(o_acc[qi].x, pv[0], vlo01);
                ggml_cuda_mad(o_acc[qi].y, pv[0], vhi01);
                ggml_cuda_mad(o_acc[qi].x, pv[1], vlo23);
                ggml_cuda_mad(o_acc[qi].y, pv[1], vhi23);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int qi = 0; qi < 8; ++qi) {
        const int q_local = warp*8 + qi;
        if (q_local >= qn) {
            break;
        }
        const int t = q0 + q_local;
        const float m_final = fmaxf(m_run[qi], 0.0f);
        const float scale_o = m_run[qi] > -INFINITY ? expf(m_run[qi] - m_final) : 0.0f;
        const float l_final = l_run[qi]*scale_o + expf(0.0f - m_final);
        const float inv_l   = scale_o / l_final;
        float * out_t = dst + int64_t(t)*s_dst1 + h*D;
        const float * gt = gate ? gate + int64_t(t)*s_g1 + h*D : nullptr;
        float v0 = o_acc[qi].x*inv_l;
        float v1 = o_acc[qi].y*inv_l;
        if (gt) {
            v0 *= 1.0f/(1.0f + expf(-gt[2*lane + 0]));
            v1 *= 1.0f/(1.0f + expf(-gt[2*lane + 1]));
        }
        out_t[2*lane + 0] = v0;
        out_t[2*lane + 1] = v1;
    }
}

// GQA tiled causal prefill attention with 1D positions (full or standard SWA), D = 128, n_gqa = 3, 64-lane waves (HIP only).
// One block = one KV head x 32 queries x its 3 query heads = 96 rows; wave w of 8 owns queries 4w..4w+3 of all 3 heads (12 rows),
// so the 3 heads share every K/V tile and the bias table row of each (query, key).
// Both matmuls are bound by LDS reads (every b128 wave read costs ~8 CU clocks on gfx906, broadcast or not, i.e. 128 B/clk),
// so each lane owns a register block: the two half-waves are row groups of 6 rows (2 queries x 3 heads); QK: lane = 6 rows x
// keys kl, kl + 32 (8 reads per 48 dot2); PV: lane = 6 rows x dims 4*kl .. 4*kl + 3 (10 reads per 96 dot2), V stored key-pair
// interleaved so one v_dot2 covers 2 keys. Softmax in log2 units with a lazy per-row stabilizer m: a row is reduced over its
// group and rescaled only when one of its logits exceeds m by more than INKLING2_PF_TAU (so p <= 2^tau fits the FP16 P tile).
// Tile classes: skip (nothing visible to the block), fast (per wave: every valid key visible to its 4 queries and beyond the
// bias extent), slow (per element mask + relative bias). The next tile's K/V bytes are prefetched into registers one phase ahead.
// v_dot2_f32_f16 as a builtin: ggml_cuda_mad emits it as asm volatile, which the scheduler cannot move LDS reads across
static __device__ __forceinline__ void inkling2_dot2(float & acc, const half2 a, const half2 b) {
#if defined(GGML_USE_HIP) && defined(V_DOT2_F32_F16_AVAILABLE)
    typedef _Float16 f16x2 __attribute__((ext_vector_type(2)));
    acc = __builtin_amdgcn_fdot2(__builtin_bit_cast(f16x2, a), __builtin_bit_cast(f16x2, b), acc, false);
#else
    ggml_cuda_mad(acc, a, b);
#endif
}

static constexpr float INKLING2_PF_TAU = 8.0f;

// block barrier that only orders LDS: LLVM puts s_waitcnt vmcnt(0) before every s_barrier on gfx906 (no back-off
// barrier), which would wait for the K/V prefetch loads still in flight; outstanding global loads to registers are
// tracked per wave and waited for at their first use as usual
static __device__ __forceinline__ void inkling2_lds_barrier() {
#if defined(GGML_USE_HIP)
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
#else
    __syncthreads();
#endif
}

template <bool kv_q8_0>
static __global__ void __launch_bounds__(512, 1) inkling2_attn_prefill_gqa(
        const float * __restrict__ q, const char * __restrict__ k, const char * __restrict__ v,
        const float * __restrict__ table, const float * __restrict__ r, const float * __restrict__ sinks,
        const int32_t * __restrict__ cell_pos, const int32_t * __restrict__ q_pos,
        const float * __restrict__ gate, float * __restrict__ dst, float * __restrict__ partial,
        const int T, const int n_kv, const int n_rel, const int n_extent, const int n_swa,
        const int64_t s_q1, const int64_t s_q2, const int s_k1, const int s_k2, const int s_v1, const int s_v2,
        const uint32_t nb_k, const uint32_t nb_v, const int64_t s_t1, const int64_t s_r1, const int64_t s_r2,
        const int64_t s_s1, const int64_t s_g1, const int64_t s_dst1, const float scale) {
#if defined(GGML_USE_HIP)
    constexpr int   D     = 128;
    constexpr int   G     = 3;
    constexpr int   BQ    = 32;
    constexpr int   BK    = 64;
    constexpr int   NW    = 8;      // waves: 2 per SIMD (1 wave/SIMD issues dot2 at only ~70% of peak on gfx906)
    constexpr int   NT    = 64*NW;
    constexpr int   NR    = BQ*G;   // 96 rows, row = qi*G + g
    constexpr int   RW    = NR/NW;  // 12 rows per wave
    constexpr int   QW    = BQ/NW;  // 4 queries per wave
    constexpr float LOG2E = 1.4426950408889634f;

    const int q0   = blockIdx.x*BQ;
    const int qn   = min(BQ, T - q0);
    const int hk   = blockIdx.y;
    const int tid  = threadIdx.x;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 6);

    // ~59 KiB: 1 block (8 waves) per CU, <= 128 VGPRs per lane
    __shared__ __align__(16) half2   s_Q[NR][D/2];    // scale*log2e folded in
    __shared__ __align__(16) half2   s_KV[BK*D/2];    // K: [key][16 B chunk ^ (key & 15)]; V: [key pair][dim] half2 of 2 keys
    __shared__ __align__(16) half    s_P[NR][BK];
    __shared__ __align__(16) float   s_R[NR][16];     // log2e folded in, zero padded to 16
    __shared__               float   s_sink[NR];      // log2e folded in
    __shared__               float   s_cs[NR];        // per-row stabilizer of u = logit + sink (log2 units), wave-uniform
    __shared__               int32_t s_qpos[BQ];
    __shared__               int32_t s_vis[2][NW];     // per-wave "tile visible" votes, double buffered by tile parity
    __shared__               int32_t s_cpos[BK];       // cell positions of the current tile (-1 = invalid), from wave 0

    const float qscale = scale*LOG2E;
    for (int idx = tid; idx < NR*(D/2); idx += NT) {
        const int row = idx/(D/2);
        const int j   = idx%(D/2);
        const int qi  = row/G;
        half2 val = make_half2(0.0f, 0.0f);
        if (qi < qn) {
            const float * qp = q + int64_t(q0 + qi)*s_q2 + int64_t(hk*G + row%G)*s_q1;
            val = make_half2(qp[2*j]*qscale, qp[2*j + 1]*qscale);
        }
        s_Q[row][j] = val;
    }
    for (int idx = tid; idx < NR*16; idx += NT) {
        const int row = idx/16;
        const int rr  = idx%16;
        const int qi  = row/G;
        s_R[row][rr] = qi < qn && rr < n_rel ? LOG2E*r[int64_t(q0 + qi)*s_r2 + int64_t(hk*G + row%G)*s_r1 + rr] : 0.0f;
    }
    if (tid < NR) {
        const int qi = tid/G;
        s_sink[tid] = qi < qn ? LOG2E*sinks[int64_t(q0 + qi)*s_s1 + hk*G + tid%G] : 0.0f;
        s_cs[tid]   = -INFINITY;
    }
    if (tid < BQ) {
        s_qpos[tid] = tid < qn ? q_pos[q0 + tid] : 0;
    }
    __syncthreads();

    // wave-uniform: kept in SGPRs
    int32_t p_min = INT32_MAX;
    int32_t p_max = INT32_MIN;
    for (int qi = 0; qi < qn; ++qi) {
        p_min = min(p_min, s_qpos[qi]);
        p_max = max(p_max, s_qpos[qi]);
    }
    p_min = __builtin_amdgcn_readfirstlane(p_min);
    p_max = __builtin_amdgcn_readfirstlane(p_max);
    int32_t w_min = INT32_MAX;
    int32_t w_max = INT32_MIN;
    for (int ql = 0; ql < QW; ++ql) {
        if (wave*QW + ql < qn) {
            w_min = min(w_min, s_qpos[wave*QW + ql]);
            w_max = max(w_max, s_qpos[wave*QW + ql]);
        }
    }
    w_min = __builtin_amdgcn_readfirstlane(w_min);
    w_max = __builtin_amdgcn_readfirstlane(w_max);
    const bool w_full = wave*QW + QW <= qn;

    const int n_active = min(n_kv, max(0, cell_pos[n_kv]));
    // buffer loads: SGPR base + 32-bit VGPR offset (the host checks that K and V fit in 4 GiB)
    typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
    typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
    const __amdgpu_buffer_rsrc_t rk = __builtin_amdgcn_make_buffer_rsrc((void *) k, (short) 0, nb_k, 0x00027000);
    const __amdgpu_buffer_rsrc_t rv = __builtin_amdgcn_make_buffer_rsrc((void *) v, (short) 0, nb_v, 0x00027000);
    const __amdgpu_buffer_rsrc_t rc = __builtin_amdgcn_make_buffer_rsrc((void *) cell_pos, (short) 0, 4*(n_kv + 1), 0x00027000);
    auto ld128 = [](const __amdgpu_buffer_rsrc_t r, const uint32_t off) {
        const u32x4 t = __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, 0);
        return make_uint4(t.x, t.y, t.z, t.w);
    };
    const uint32_t k_head = uint32_t(hk)*s_k1;
    const uint32_t v_head = uint32_t(hk)*s_v1;

    // lane-derived indices are recomputed at each use site from an opaque copy of tid: otherwise LLVM hoists ~15 LDS and
    // global addresses out of the tile loop and spills them, and the vmcnt(0) of every scratch reload also waits for the
    // K/V prefetch loads in flight
    auto otid = [&]() {
        int t = tid;
        asm volatile("" : "+v"(t));
        return t;
    };

    // loaders: K thread = (key tid/8, 16 dims (tid%8)*16); V thread = (key pair tid/16, 8 dims (tid%16)*8)
    uint4   kr[kv_q8_0 ? 1 : 2];
    float   kd  = 0.0f;
    int32_t lcp = -1; // cell position of key k0 + lane (QK); the loaders read theirs from s_cpos
    uint4   vr[2];    // q8_0: .x/.y = 8 quants; f16: 8 halves
    float   vd[2]  = {0.0f, 0.0f};

    // loads are unconditional from a clamped cell (no phis across the tile loop); validity travels in the cell position
    const int k_last = max(n_active - 1, 0);
    auto load_k = [&](const int k0) {
        const int t   = otid();
        const int kk  = t >> 3;
        const int kb  = t & 7;
        const int ln  = t & 63;
        const int key = min(k0 + kk, k_last);
        const int kl  = min(k0 + ln, k_last);
        const int32_t c_l = __builtin_amdgcn_raw_buffer_load_b32(rc, 4*kl, 0, 0);
        lcp = k0 + ln < n_active ? c_l : -1;
        const uint32_t row = k_head + uint32_t(key)*s_k2;
        if constexpr (kv_q8_0) {
            const uint32_t blk = row + (kb >> 1)*sizeof(block_q8_0);
            kd    = __half2float(__ushort_as_half(__builtin_amdgcn_raw_buffer_load_b16(rk, blk, 0, 0)));
            kr[0] = ld128(rk, blk + 2 + 16*(kb & 1));
        } else {
            kr[0] = ld128(rk, row + 32*kb + 0);
            kr[1] = ld128(rk, row + 32*kb + 16);
        }
    };
    auto store_k = [&]() {
        const int t  = otid();
        const int kk = t >> 3;
        const int kb = t & 7;
        uint4 h4[2];
        const bool kcp = s_cpos[kk] >= 0;
        if constexpr (kv_q8_0) {
            const int * qw = (const int *) kr;
            half2 * h = (half2 *) h4;
            const float d = kcp ? kd : 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int w = kcp ? qw[j] : 0;
                h[2*j + 0] = make_half2(d*int8_t(w >>  0), d*int8_t(w >>  8));
                h[2*j + 1] = make_half2(d*int8_t(w >> 16), d*int8_t(w >> 24));
            }
        } else {
            h4[0] = kcp ? kr[0] : make_uint4(0, 0, 0, 0);
            h4[1] = kcp ? kr[1] : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int sc = (2*kb + c) ^ (kk & 15);
            *(uint4 *) &s_KV[kk*(D/2) + 4*sc] = h4[c];
        }
    };
    auto load_v = [&](const int k0) {
        const int t  = otid();
        const int kp = t >> 4;
        const int vs = t & 15;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int key = min(k0 + 2*kp + e, k_last);
            const uint32_t row = v_head + uint32_t(key)*s_v2;
            if constexpr (kv_q8_0) {
                const uint32_t blk = row + (vs >> 2)*sizeof(block_q8_0);
                vd[e] = __half2float(__ushort_as_half(__builtin_amdgcn_raw_buffer_load_b16(rv, blk, 0, 0)));
                const u32x2 t = __builtin_amdgcn_raw_buffer_load_b64(rv, blk + 2 + 8*(vs & 3), 0, 0);
                vr[e].x = t.x;
                vr[e].y = t.y;
            } else {
                vr[e] = ld128(rv, row + 16*vs);
            }
        }
    };
    auto store_v = [&]() {
        const int t  = otid();
        const int kp = t >> 4;
        const int vs = t & 15;
        uint4 h4[2];
        half2 * h = (half2 *) h4;
        const int32_t vcp[2] = {s_cpos[2*kp + 0], s_cpos[2*kp + 1]};
        if constexpr (kv_q8_0) {
            const float d0 = vcp[0] >= 0 ? vd[0] : 0.0f;
            const float d1 = vcp[1] >= 0 ? vd[1] : 0.0f;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int a = vcp[0] >= 0 ? (j ? vr[0].y : vr[0].x) : 0;
                const int b = vcp[1] >= 0 ? (j ? vr[1].y : vr[1].x) : 0;
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    h[4*j + s] = make_half2(d0*int8_t(a >> (8*s)), d1*int8_t(b >> (8*s)));
                }
            }
        } else {
            const half * a = (const half *) &vr[0];
            const half * b = (const half *) &vr[1];
            const half z = __float2half(0.0f);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                h[i] = __halves2half2(vcp[0] >= 0 ? a[i] : z, vcp[1] >= 0 ? b[i] : z);
            }
        }
        *(uint4 *) &s_KV[kp*D + 8*vs + 0] = h4[0];
        *(uint4 *) &s_KV[kp*D + 8*vs + 4] = h4[1];
    };

    // QK / softmax layout: lane = (row group g, keys kl and kl + 32); group g owns rows g*RG.. of the wave, i.e. queries
    // wave*QW + g*QG.. with their 3 heads. 6 rows x 2 keys per lane need 8 b128 LDS reads per 48 dot2 instead of the
    // 13 of 12 rows x 1 key (every b128 wave read costs ~8 CU clocks on gfx906, broadcast or not)
    constexpr int NG = 2;
    constexpr int RG = RW/NG;  // 6 rows
    constexpr int QG = QW/NG;  // 2 queries

    // PV layout: lane = (the same row group g, dims 4*kl .. 4*kl + 3), so a row's O, l and stabilizer stay in its group
    float o[RG][4];
    float l_p[RG];             // per lane partial row sums of the group's rows
#pragma unroll
    for (int i = 0; i < RG; ++i) {
        o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
        l_p[i] = 0.0f;
    }

    // lazy online softmax of this lane's 2 keys for the group's rows (u: logits without the sink, -inf = masked);
    // the stabilizers cs live in LDS, are read once per tile and only written back when a row is rescaled
    auto softmax_rows = [&](float (&u)[RG][2]) {
        const int t  = otid();
        const int g  = (t >> 5) & 1;
        const int kl = t & 31;
        float cs[RG];
        bool  need = false;
#pragma unroll
        for (int i = 0; i < RG; ++i) {
            cs[i] = s_cs[wave*RW + g*RG + i];
            need |= fmaxf(u[i][0], u[i][1]) > cs[i] + INKLING2_PF_TAU;
        }
        if (inkling2_wave_any(need)) {
            // rare: O is rescaled once for all rows (alpha = 1 for the rows that keep their stabilizer), so the
            // accumulators see a single conditional region instead of one per row
            float alpha[RG];
#pragma unroll
            for (int i = 0; i < RG; ++i) {
                alpha[i] = 1.0f;
                const float um = fmaxf(u[i][0], u[i][1]);
                if (inkling2_wave_any(um > cs[i] + INKLING2_PF_TAU)) {
                    // per group: the other group's row may have nothing above its stabilizer (alpha = 1 then)
                    const float cs_new = fmaxf(cs[i], inkling2_max32(um));
                    alpha[i] = cs_new > cs[i] ? __builtin_amdgcn_exp2f(cs[i] - cs_new) : 1.0f;
                    cs[i]    = cs_new;
                    if (kl == 0) {
                        s_cs[wave*RW + g*RG + i] = cs_new;
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < RG; ++i) {
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    o[i][d] *= alpha[i];
                }
                l_p[i] *= alpha[i];
            }
        }
#pragma unroll
        for (int i = 0; i < RG; ++i) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const float p = u[i][j] > -INFINITY ? __builtin_amdgcn_exp2f(u[i][j] - cs[i]) : 0.0f;
                l_p[i] += p;
                s_P[wave*RW + g*RG + i][kl + 32*j] = __float2half(p);
            }
        }
    };

    const int32_t p_lo = n_swa > 0 ? p_min - n_swa + 1 : INT32_MIN;

    // KV split (blockIdx.z of gridDim.z): a contiguous range of 64-cell tiles; partial results go to `partial`
    const int n_tiles = (n_active + BK - 1)/BK;
    const int k_begin = BK*((blockIdx.z*n_tiles)/gridDim.z);
    const int k_end   = min(n_active, BK*(((blockIdx.z + 1)*n_tiles)/gridDim.z));

    load_k(k_begin);
    for (int k0 = k_begin; k0 < k_end; k0 += BK) {
        // A: skip test (also orders the previous PV before the K/V tile is overwritten), K tile, V prefetch
        const bool vis = inkling2_wave_any(lcp >= 0 && lcp <= p_max && lcp >= p_lo);
        const int  par = (k0/BK) & 1;
        if ((otid() & 63) == 0) {
            s_vis[par][wave] = vis;
        }
        if (wave == 0) {
            s_cpos[otid() & 63] = lcp; // the previous tile's last reader (store_v) is behind the barrier of phase C
        }
        inkling2_lds_barrier();
        bool any_vis = false;
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            any_vis |= s_vis[par][w] != 0;
        }
        if (any_vis) {
        store_k();
        load_v(k0);
        inkling2_lds_barrier();

        // B: S = Q K^T for 6 rows x 2 keys per lane
        int g  = (otid() >> 5) & 1;
        int kl = otid() & 31;
        float s[RG][2];
#pragma unroll
        for (int i = 0; i < RG; ++i) {
            s[i][0] = 0.0f;
            s[i][1] = 0.0f;
        }
        {
            // software pipeline pinned with sched_barrier: Q rows PF ahead and the next chunk's K one chunk ahead
            // (the scheduler otherwise sinks every LDS read next to its use and waits lgkmcnt(0) for it)
            constexpr int PF = 3;
            static_assert(RG % PF == 0, "Q buffer rotation must repeat per chunk");
            uint4 kq[2][2];
            uint4 qq[PF];
            auto rd_k = [&](const int c, const int j) {
                return *(const uint4 *) &s_KV[(kl + 32*j)*(D/2) + 4*((c & (D/8 - 1)) ^ (kl & 15))];
            };
            auto rd_q = [&](const int i, const int c) {
                return *(const uint4 *) &s_Q[wave*RW + g*RG + i][4*(c & (D/8 - 1))];
            };
            kq[0][0] = rd_k(0, 0);
            kq[0][1] = rd_k(0, 1);
#pragma unroll
            for (int j = 0; j < PF; ++j) {
                qq[j] = rd_q(j, 0);
            }
            auto chunk = [&](const int c, const uint4 (&kc)[2], uint4 (&kn)[2]) {
#pragma unroll
                for (int i = 0; i < RG; ++i) {
                    __builtin_amdgcn_sched_barrier(0);
                    if (i == 0) {
                        kn[0] = rd_k(c + 1, 0);
                        kn[1] = rd_k(c + 1, 1);
                    }
                    const half2 * qh = (const half2 *) &qq[i % PF];
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const half2 * kh = (const half2 *) &kc[j];
                        inkling2_dot2(s[i][j], qh[0], kh[0]);
                        inkling2_dot2(s[i][j], qh[1], kh[1]);
                        inkling2_dot2(s[i][j], qh[2], kh[2]);
                        inkling2_dot2(s[i][j], qh[3], kh[3]);
                    }
                    qq[i % PF] = i + PF < RG ? rd_q(i + PF, c) : rd_q(i + PF - RG, c + 1);
                }
                __builtin_amdgcn_sched_barrier(0);
            };
#pragma unroll 1
            for (int c = 0; c < D/8; c += 2) {
                chunk(c + 0, kq[0], kq[1]);
                chunk(c + 1, kq[1], kq[0]);
            }
        }

        // fast: every valid cell is visible to all queries of the wave and beyond the bias extent (holes are masked)
        const int32_t cpk[2] = {s_cpos[kl], s_cpos[kl + 32]};
        bool fast_l = true;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int32_t cp = cpk[j];
            fast_l &= cp < 0 || (cp <= w_min && (n_swa == 0 || w_max - cp < n_swa) && w_min - cp >= n_extent);
        }
        const bool fast = w_full && inkling2_wave_all(fast_l);
        if (fast) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                if (cpk[j] < 0) {
#pragma unroll
                    for (int i = 0; i < RG; ++i) {
                        s[i][j] = -INFINITY;
                    }
                }
            }
        } else {
#pragma unroll
            for (int ql = 0; ql < QG; ++ql) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int     qi   = wave*QW + g*QG + ql;
                const int32_t qp   = s_qpos[qi];
                const int32_t cp   = cpk[j];
                const bool    ok   = qi < qn && cp >= 0 && cp <= qp && (n_swa == 0 || qp - cp < n_swa);
                const int32_t dist = qp - cp;
                float b[G] = {0.0f, 0.0f, 0.0f};
                if (ok && dist < n_extent) {
                    const float * tb = table + int64_t(dist)*s_t1;
                    if (n_rel == 16 && (s_t1 & 3) == 0) {
#pragma unroll
                        for (int c4 = 0; c4 < 4; ++c4) {
                            const float4 a = ((const float4 *) tb)[c4];
#pragma unroll
                            for (int h = 0; h < G; ++h) {
                                const float4 rr = ((const float4 *) s_R[qi*G + h])[c4];
                                b[h] += a.x*rr.x + a.y*rr.y + a.z*rr.z + a.w*rr.w;
                            }
                        }
                    } else {
                        for (int rr = 0; rr < n_rel; ++rr) {
                            const float a = tb[rr];
#pragma unroll
                            for (int h = 0; h < G; ++h) {
                                b[h] += a*s_R[qi*G + h][rr];
                            }
                        }
                    }
                }
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    s[ql*G + h][j] = ok ? s[ql*G + h][j] + b[h] : -INFINITY;
                }
                __builtin_amdgcn_sched_barrier(0); // one (query, key) table row in flight at a time
            }
            }
        }
        softmax_rows(s);

        // C: V tile (key-pair interleaved) over the consumed K tile, next K prefetch (hidden behind PV)
        inkling2_lds_barrier();
        store_v();
        load_k(k0 + BK);
        inkling2_lds_barrier();

        // D: O += P V for 6 rows x 4 dims per lane: 6 P + 4 V b128 reads per 96 dot2
        {
            g  = (otid() >> 5) & 1;
            kl = otid() & 31;
            // same pinned pipeline as QK: P rows PF ahead, the next 8 keys' V one step ahead
            constexpr int PF = 3;
            uint4 vv[2][4];
            uint4 pp[PF];
            auto rd_v = [&](const int k8, uint4 (&vt)[4]) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    vt[j] = *(const uint4 *) &s_KV[(4*(k8 & (BK/8 - 1)) + j)*D + 4*kl];
                }
            };
            auto rd_p = [&](const int i, const int k8) {
                return *(const uint4 *) &s_P[wave*RW + g*RG + i][8*(k8 & (BK/8 - 1))];
            };
            rd_v(0, vv[0]);
#pragma unroll
            for (int j = 0; j < PF; ++j) {
                pp[j] = rd_p(j, 0);
            }
            auto step = [&](const int k8, const uint4 (&vc)[4], uint4 (&vn)[4]) {
#pragma unroll
                for (int i = 0; i < RG; ++i) {
                    __builtin_amdgcn_sched_barrier(0);
                    if (i == 0) {
                        rd_v(k8 + 1, vn);
                    }
                    const half2 * ph = (const half2 *) &pp[i % PF];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const half2 * vh = (const half2 *) &vc[j];
#pragma unroll
                        for (int d = 0; d < 4; ++d) {
                            inkling2_dot2(o[i][d], ph[j], vh[d]);
                        }
                    }
                    pp[i % PF] = i + PF < RG ? rd_p(i + PF, k8) : rd_p(i + PF - RG, k8 + 1);
                }
                __builtin_amdgcn_sched_barrier(0);
            };
#pragma unroll 1
            for (int k8 = 0; k8 < BK/8; k8 += 2) {
                step(k8 + 0, vv[0], vv[1]);
                step(k8 + 1, vv[1], vv[0]);
            }
        }
        } else {
            load_k(k0 + BK);
        }
    }

    const int g  = (otid() >> 5) & 1;
    const int kl = otid() & 31;
#pragma unroll
    for (int i = 0; i < RG; ++i) {
        const float l  = inkling2_sum32(l_p[i]);
        const int   r  = g*RG + i;  // row of the wave
        const int   qi = wave*QW + r/G;
        if (qi >= qn) {
            continue;
        }
        const int   t       = q0 + qi;
        const int   h       = hk*G + r%G;
        const float m       = s_cs[wave*RW + r] - s_sink[wave*RW + r]; // stabilizer of the logits (sink included)
        if (partial) {
            // [split][t][h][D + 2]: unnormalized O, then m and l
            float * pt = partial + ((int64_t(blockIdx.z)*T + t)*(gridDim.y*G) + h)*(D + 2);
            *(float4 *) (pt + 4*kl) = make_float4(o[i][0], o[i][1], o[i][2], o[i][3]);
            if (kl == 0) {
                pt[D]     = m;
                pt[D + 1] = l;
            }
            continue;
        }
        const float m_final = fmaxf(m, 0.0f);
        const float scale_o = m > -INFINITY ? __builtin_amdgcn_exp2f(m - m_final) : 0.0f;
        const float l_final = l*scale_o + __builtin_amdgcn_exp2f(-m_final);
        const float inv_l   = scale_o/l_final;
        float v[4];
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            v[d] = o[i][d]*inv_l;
        }
        if (gate) {
            const float * gt = gate + int64_t(t)*s_g1 + h*D + 4*kl;
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                v[d] *= 1.0f/(1.0f + expf(-gt[d]));
            }
        }
        *(float4 *) (dst + int64_t(t)*s_dst1 + h*D + 4*kl) = make_float4(v[0], v[1], v[2], v[3]);
    }
#else
    GGML_UNUSED_VARS(q, k, v, table, r, sinks, cell_pos, q_pos, gate, dst, T, n_kv, n_rel, n_extent, n_swa,
        s_q1, s_q2, s_k1, s_k2, s_v1, s_v2, s_t1, s_r1, s_r2, s_s1, s_g1, s_dst1, scale);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP)
}

// merges the KV splits of inkling2_attn_prefill_gqa (log2-unit stabilizers) and applies the zero sink and the gate;
// one 64-lane wave per (token, head), lane = dim pair
static __global__ void __launch_bounds__(256) inkling2_attn_prefill_combine(
        const float * __restrict__ partial, const float * __restrict__ gate, float * __restrict__ dst,
        const int T, const int n_head, const int n_split, const int64_t s_g1, const int64_t s_dst1) {
    constexpr int D = 128;
    const int task = blockIdx.x*4 + threadIdx.x/64;
    const int lane = threadIdx.x % 64;
    if (task >= T*n_head) {
        return;
    }
    const int t = task / n_head;
    const int h = task % n_head;

    float M = -INFINITY;
    for (int sp = 0; sp < n_split; ++sp) {
        M = fmaxf(M, partial[((int64_t(sp)*T + t)*n_head + h)*(D + 2) + D]);
    }
    float  l = 0.0f;
    float2 o = make_float2(0.0f, 0.0f);
    if (M > -INFINITY) {
        for (int sp = 0; sp < n_split; ++sp) {
            const float * pt = partial + ((int64_t(sp)*T + t)*n_head + h)*(D + 2);
            const float   ms = pt[D];
            if (ms > -INFINITY) {
                const float  w  = exp2f(ms - M);
                const float2 os = *(const float2 *) (pt + 2*lane);
                o.x += w*os.x;
                o.y += w*os.y;
                l   += w*pt[D + 1];
            }
        }
    }
    const float m_final = fmaxf(M, 0.0f);
    const float scale_o = M > -INFINITY ? exp2f(M - m_final) : 0.0f;
    const float l_final = l*scale_o + exp2f(-m_final);
    const float inv_l   = scale_o/l_final;
    float v0 = o.x*inv_l;
    float v1 = o.y*inv_l;
    if (gate) {
        const float * gt = gate + int64_t(t)*s_g1 + h*D;
        v0 *= 1.0f/(1.0f + expf(-gt[2*lane + 0]));
        v1 *= 1.0f/(1.0f + expf(-gt[2*lane + 1]));
    }
    *(float2 *) (dst + int64_t(t)*s_dst1 + h*D + 2*lane) = make_float2(v0, v1);
}

bool ggml_cuda_inkling2_attn_supported(const ggml_tensor * op) {
    const ggml_type type    = op->src[1]->type;
    const int64_t   d       = op->src[0]->ne[0];
    const bool      v_trans = ggml_get_op_params_i32(op, 1) != 0;
    const bool      compact = op->src[7]->type == GGML_TYPE_I32;
    if (compact) {
        const int32_t n_rot_q = ggml_get_op_params_i32(op, 2);
        const int32_t n_rot_v = ggml_get_op_params_i32(op, 3);
        return !v_trans && d == 128 && op->src[3]->type == GGML_TYPE_F32 && op->src[3]->ne[0] <= 16 &&
            (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_F16) && n_rot_q == 0 && n_rot_v == 0 &&
            ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size == 64;
    }
    const bool ok_type = type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 ||
        (type == GGML_TYPE_Q8_0 && !v_trans && d % QK8_0 == 0);
    return ok_type && op->src[3]->type == GGML_TYPE_F32 && d <= 512;
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
    const int T      = q->ne[2];
    const int n_kv   = k->ne[2];
    const int n_rel  = table->ne[0];
    const int n_gqa  = n_head/k->ne[1];

    if (mask->type == GGML_TYPE_I32) {
        const int n_swa = ggml_get_op_params_i32(dst, 4);
        const int64_t s_g1 = gate ? gate->nb[1]/sizeof(float) : 0;
        // the GQA kernel addresses K and V with 32-bit buffer offsets
        if (n_gqa == 3 && ggml_nbytes(k) < (size_t(1) << 31) && ggml_nbytes(v) < (size_t(1) << 31)) {
            // KV split so that the (1 block per CU) grid fills whole rounds: 256 blocks on 60 CUs take 5 rounds unsplit,
            // 13 rounds of 1/3 the work with 3 splits; only full-attention layers (SWA visibility clusters in a few tiles)
            const int n_blocks = ((T + 31)/32)*k->ne[1];
            const int n_sm     = ggml_cuda_info().devices[ctx.device].nsm;
            const int s_max    = n_swa > 0 ? 1 : std::min(n_blocks >= n_sm ? 4 : 16, std::max(1, n_kv/2048));
            int    n_split = 1;
            double best    = (n_blocks + n_sm - 1)/n_sm;
            for (int sp = 2; sp <= s_max; ++sp) {
                const double rounds = double((n_blocks*sp + n_sm - 1)/n_sm)/sp;
                if (rounds < best - 1e-9) {
                    best    = rounds;
                    n_split = sp;
                }
            }
            ggml_cuda_pool_alloc<float> partial(ctx.pool());
            if (n_split > 1) {
                partial.alloc((size_t) n_split*T*n_head*(d + 2));
            }
            const dim3 grid_gqa((T + 31)/32, k->ne[1], n_split);
            auto launch_gqa = [&](auto kv_tag) {
                inkling2_attn_prefill_gqa<decltype(kv_tag)::value><<<grid_gqa, 512, 0, ctx.stream()>>>(
                    (const float *) q->data, (const char *) k->data, (const char *) v->data,
                    (const float *) table->data, (const float *) r->data, (const float *) sinks->data,
                    (const int32_t *) rel_pos->data, (const int32_t *) mask->data,
                    gate ? (const float *) gate->data : nullptr, (float *) dst->data, n_split > 1 ? partial.get() : nullptr,
                    T, n_kv, n_rel, (int) table->ne[1], n_swa,
                    q->nb[1]/sizeof(float), q->nb[2]/sizeof(float), (int) k->nb[1], (int) k->nb[2],
                    (int) v->nb[1], (int) v->nb[2], (uint32_t) ggml_nbytes(k), (uint32_t) ggml_nbytes(v),
                    table->nb[1]/sizeof(float), r->nb[1]/sizeof(float), r->nb[2]/sizeof(float),
                    sinks->nb[1]/sizeof(float), s_g1, dst->nb[1]/sizeof(float), scale);
            };
            if (k->type == GGML_TYPE_Q8_0) {
                launch_gqa(std::true_type{});
            } else {
                launch_gqa(std::false_type{});
            }
            if (n_split > 1) {
                inkling2_attn_prefill_combine<<<(T*n_head + 3)/4, 256, 0, ctx.stream()>>>(
                    partial.get(), gate ? (const float *) gate->data : nullptr, (float *) dst->data,
                    T, n_head, n_split, s_g1, dst->nb[1]/sizeof(float));
            }
            return;
        }
        const dim3 grid((T + 31)/32, n_head, 1);
        if (k->type == GGML_TYPE_Q8_0) {
            inkling2_attn_prefill_tile<true><<<grid, 256, 0, ctx.stream()>>>(
                (const float *) q->data, (const char *) k->data, (const char *) v->data,
                (const float *) table->data, (const float *) r->data, (const float *) sinks->data,
                (const int32_t *) rel_pos->data, (const int32_t *) mask->data,
                gate ? (const float *) gate->data : nullptr, (float *) dst->data,
                T, n_head, n_gqa, n_kv, n_rel, (int) table->ne[1], n_swa,
                q->nb[1]/sizeof(float), q->nb[2]/sizeof(float), k->nb[1], k->nb[2],
                v->nb[1], v->nb[2], table->nb[1]/sizeof(float), r->nb[1]/sizeof(float), r->nb[2]/sizeof(float),
                sinks->nb[1]/sizeof(float), s_g1, dst->nb[1]/sizeof(float), scale);
        } else {
            inkling2_attn_prefill_tile<false><<<grid, 256, 0, ctx.stream()>>>(
                (const float *) q->data, (const char *) k->data, (const char *) v->data,
                (const float *) table->data, (const float *) r->data, (const float *) sinks->data,
                (const int32_t *) rel_pos->data, (const int32_t *) mask->data,
                gate ? (const float *) gate->data : nullptr, (float *) dst->data,
                T, n_head, n_gqa, n_kv, n_rel, (int) table->ne[1], n_swa,
                q->nb[1]/sizeof(float), q->nb[2]/sizeof(float), k->nb[1], k->nb[2],
                v->nb[1], v->nb[2], table->nb[1]/sizeof(float), r->nb[1]/sizeof(float), r->nb[2]/sizeof(float),
                sinks->nb[1]/sizeof(float), s_g1, dst->nb[1]/sizeof(float), scale);
        }
        return;
    }

    if (k->type == GGML_TYPE_Q8_0) {
        if (n_kv > 256 && n_gqa >= 2 && n_gqa <= 4 && d == 128 && table->ne[0] <= 16 &&
                ggml_cuda_info().devices[ctx.device].warp_size == 64) {
            // snap n_splits so n_hk_grp*n_splits fills 1 full wave of 3 blocks/CU (2 waves at > 128k) with no tail
            const int n_head_kv = k->ne[1];
            const int n_sm      = ggml_cuda_info().devices[ctx.device].nsm;
            const int n_hk_grp  = (n_head_kv + INKLING2_ATTN_WAVES - 1) / INKLING2_ATTN_WAVES;
            const int n_waves_g = n_kv > 131072 ? 2 : 1;
            const int target_splits = std::min(INKLING2_ATTN_MAX_SPLITS, std::max(1, (3*n_waves_g*n_sm) / n_hk_grp));
            int n_splits = std::min((n_kv + 63) / 64, target_splits);
            ggml_cuda_pool_alloc<float> partial(ctx.pool(), (size_t) n_head * n_splits * (d + 2));

            auto launch_gqa = [&](auto mask_tag, auto g_tag) {
                using M = decltype(mask_tag);
                constexpr int G = decltype(g_tag)::value;
                inkling2_attn_q8_0_split_gqa<M, G><<<dim3(n_hk_grp, n_splits), 64*INKLING2_ATTN_WAVES, 0, ctx.stream()>>>(
                    (const float *) q->data, (const char *) k->data, (const char *) v->data, (const float *) table->data,
                    (const float *) r->data, (const float *) sinks->data, (const int32_t *) rel_pos->data, (const M *) mask->data,
                    partial.get(), n_head_kv, n_kv, n_rel, (int) table->ne[1], q->nb[1]/sizeof(float), k->nb[1], k->nb[2], v->nb[1], v->nb[2],
                    table->nb[1]/sizeof(float), r->nb[1]/sizeof(float), scale, n_rot_q, n_splits);
            };
            auto launch_m = [&](auto mask_tag) {
                switch (n_gqa) {
                    case 2: launch_gqa(mask_tag, std::integral_constant<int, 2>()); break;
                    case 3: launch_gqa(mask_tag, std::integral_constant<int, 3>()); break;
                    case 4: launch_gqa(mask_tag, std::integral_constant<int, 4>()); break;
                    default: GGML_ABORT("unexpected n_gqa %d", n_gqa);
                }
            };
            mask->type == GGML_TYPE_F16 ? launch_m(half()) : launch_m(float());

            const int n_threads_c = std::max(512, ((d + 63) / 64) * 64);
            const int ng_c        = std::max(1, n_threads_c / d);
            inkling2_attn_combine<<<n_head, n_threads_c, (size_t) ng_c * d * sizeof(float), ctx.stream()>>>(
                partial.get(), (const float *) gate->data, (float *) dst->data, d, n_splits, n_rot_v);
            return;
        }
        if (n_kv > 256) {
            const int n_splits   = std::min(64, std::max(4, (n_kv + 127) / 128));
            const int n_threads  = d > 256 ? 512 : 256;
            const int ng_v       = n_threads / (d / 4);
            const size_t smem_s1 = (INKLING2_ATTN_SPLIT_TILE + d + n_rel + ng_v * d) * sizeof(float);
            ggml_cuda_pool_alloc<float> partial(ctx.pool(), (size_t) n_head * n_splits * (d + 2));

            auto launch_split = [&](auto mask_tag) {
                using M = decltype(mask_tag);
                inkling2_attn_q8_0_split<M><<<dim3(n_head, n_splits), n_threads, smem_s1, ctx.stream()>>>(
                    (const float *) q->data, (const char *) k->data, (const char *) v->data, (const float *) table->data,
                    (const float *) r->data, (const float *) sinks->data, (const int32_t *) rel_pos->data, (const M *) mask->data,
                    partial.get(), d, n_gqa, n_kv, n_rel, (int) table->ne[1], q->nb[1]/sizeof(float), k->nb[1], k->nb[2], v->nb[1], v->nb[2],
                    table->nb[1]/sizeof(float), r->nb[1]/sizeof(float), scale, n_rot_q, n_splits);
            };
            mask->type == GGML_TYPE_F16 ? launch_split(half()) : launch_split(float());

            const int n_threads_c = std::max(512, ((d + 63) / 64) * 64);
            const int ng_c        = std::max(1, n_threads_c / d);
            inkling2_attn_combine<<<n_head, n_threads_c, (size_t) ng_c * d * sizeof(float), ctx.stream()>>>(
                partial.get(), (const float *) gate->data, (float *) dst->data, d, n_splits, n_rot_v);
            return;
        }

        const int    n_threads = 512;
        const size_t smem      = (n_kv + d + n_rel)*sizeof(float);
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

    if (n_kv > 256) {
        const int n_splits   = std::min(64, std::max(4, (n_kv + 127) / 128));
        const int n_threads  = d > 256 ? 512 : 256;
        const int vbuf_elems = v_trans ? d : (n_threads / d) * d;
        const size_t smem_s1 = (INKLING2_ATTN_SPLIT_TILE + d + n_rel + vbuf_elems) * sizeof(float);
        ggml_cuda_pool_alloc<float> partial(ctx.pool(), (size_t) n_head * n_splits * (d + 2));

        auto launch_s1 = [&](auto kernel, const auto * k_d, const auto * v_d, const auto * mask_d) {
            kernel<<<dim3(n_head, n_splits), n_threads, smem_s1, ctx.stream()>>>(
                (const float *) q->data, k_d, v_d, (const float *) table->data, (const float *) r->data,
                (const float *) sinks->data, (const int32_t *) rel_pos->data, mask_d, partial.get(),
                d, n_gqa, n_kv, n_rel, (int) table->ne[1], q->nb[1]/sizeof(float), k->nb[1]/ts, k->nb[2]/ts, v->nb[1]/ts, v->nb[2]/ts,
                table->nb[1]/sizeof(float), r->nb[1]/sizeof(float), scale, n_rot_q, n_splits);
        };
        auto launch_st = [&](auto tag) {
            using T = decltype(tag);
            const T * k_d = (const T *) k->data;
            const T * v_d = (const T *) v->data;
            auto launch_sm = [&](auto mask_tag) {
                using M = decltype(mask_tag);
                const M * mask_d = (const M *) mask->data;
                if (v_trans) {
                    vec ? launch_s1(inkling2_attn_split_f32<T, M, true,  true>,  k_d, v_d, mask_d)
                        : launch_s1(inkling2_attn_split_f32<T, M, true,  false>, k_d, v_d, mask_d);
                } else {
                    vec ? launch_s1(inkling2_attn_split_f32<T, M, false, true>,  k_d, v_d, mask_d)
                        : launch_s1(inkling2_attn_split_f32<T, M, false, false>, k_d, v_d, mask_d);
                }
            };
            mask->type == GGML_TYPE_F16 ? launch_sm(half()) : launch_sm(float());
        };
        switch (k->type) {
            case GGML_TYPE_F32:  launch_st(float());       break;
            case GGML_TYPE_F16:  launch_st(half());        break;
            case GGML_TYPE_BF16: launch_st(nv_bfloat16()); break;
            default: GGML_ABORT("unsupported type %s", ggml_type_name(k->type));
        }

        const int n_threads_c = std::max(512, ((d + 63) / 64) * 64);
        const int ng_c        = std::max(1, n_threads_c / d);
        inkling2_attn_combine<<<n_head, n_threads_c, (size_t) ng_c * d * sizeof(float), ctx.stream()>>>(
            partial.get(), (const float *) gate->data, (float *) dst->data, d, n_splits, n_rot_v);
        return;
    }

    const int    n_threads = 512;
    const size_t smem      = (n_kv + d + n_rel)*sizeof(float);

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
