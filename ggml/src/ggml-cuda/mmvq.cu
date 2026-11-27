#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <type_traits>

// only enabled on DGX Spark, where it is a gain on every type below. On the higher-bandwidth parts the kernel
// has little exposed latency left to hide and the extra requests cost more than they save.
// For perf data, see https://github.com/ggml-org/llama.cpp/pull/26705#issuecomment-5569335031
#if __CUDA_ARCH__ == GGML_CUDA_CC_GB10
// returns true only for those quants that benefit from prefetch and false otherwise
static constexpr __host__ __device__ bool mmvq_should_prefetch(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}
#endif

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA1,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10,
    MMVQ_PARAMETERS_ORIN
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA1)
    return MMVQ_PARAMETERS_RDNA1;
#elif defined(RDNA2) || defined(RDNA3_5) || defined(RDNA4_M)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif __CUDA_ARCH__ == GGML_CUDA_CC_GB10
    return MMVQ_PARAMETERS_GB10;
#elif __CUDA_ARCH__ == GGML_CUDA_CC_ORIN
    return MMVQ_PARAMETERS_ORIN;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA1(cc)) {
        return MMVQ_PARAMETERS_RDNA1;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_GB10) {
        return MMVQ_PARAMETERS_GB10;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_ORIN) {
        return MMVQ_PARAMETERS_ORIN;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: P100, Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        cc = ggml_cuda_highest_compiled_arch(cc);
        if (cc == GGML_CUDA_CC_PASCAL || cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_GB10) {
        switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ORIN) {
        switch (type) { // tuned for Jetson Orin
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                return ne11 <= 1;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_VOLTA) {
        switch (type) {
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        switch (type) { // tuned on Radeon 8060S (gfx1151)
            case GGML_TYPE_Q8_0:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3) || defined(RDNA4_M)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif !defined(GGML_USE_MUSA) && (__CUDA_ARCH__ == GGML_CUDA_CC_PASCAL || __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
                // NVFP4 on gfx906 (Inkling 2 Mini experts): 1 warp per row is 25-30% faster than 2
                return type == GGML_TYPE_NVFP4 ? 1 : 2;
            case 2:
            case 3:
                return type == GGML_TYPE_Q3_K ? 1 : 2;
            case 4:
                return type == GGML_TYPE_Q6_K? 1 : 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA1) {
        switch (ncols_dst) {
            case 1:
                return 1;
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    if (table_id == MMVQ_PARAMETERS_ORIN) {
        return calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
    }
    return 1;
}

// Low bit types load few weight bytes per thread, a second row per block shares the q8_1 loads and adds memory parallelism.
static constexpr __host__ __device__ bool mmvq_two_rows_ncols1(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
            return true;
        default:
            return false;
    }
}

static constexpr __host__ __device__ int calc_rows_per_block(ggml_type type, int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10 || table_id == MMVQ_PARAMETERS_RDNA1) {
        switch (ncols_dst) {
            case 1:
                if (small_k) {
                    return nwarps;
                }
                if (table_id == MMVQ_PARAMETERS_GB10) {
                    return type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S ? 2 : 1;
                }
                return (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_RDNA1) && mmvq_two_rows_ncols1(type) ? 2 : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_ORIN) {
        switch (ncols_dst) {
            case 1:
                if (small_k) {
                    return nwarps;
                }
                // One row per warp (more rows per block, more loads in flight) for these types, measured on Jetson Orin Nano.
                // Other types are faster in isolated mat-vec too, but gained at most 2 % end to end.
                switch (type) {
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ4_NL:
                        return nwarps;
                    default:
                        break;
                }
                return mmvq_two_rows_ncols1(type) ? 2 : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = shared_expert ? 0 : ncols_dst == 1 && ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#if __CUDA_ARCH__ == GGML_CUDA_CC_GB10
        // start the next iterations' weight loads early
        if constexpr (mmvq_should_prefetch(type)) {
            // loop iterations, not blocks; q3_K and q6_K keep the distance in blocks they had with vdr 1
            constexpr int pf_dist = type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q6_K ? 1 : 2;
            const int kbx_pf = kbx + pf_dist*blocks_per_iter;
            if (kbx_pf < blocks_per_row_x) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    const size_t off = (size_t)(kbx_offset + i*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
                    ggml_cuda_prefetch_l2((const char *) vx + off);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            ggml_cuda_prefetch_l2((const char *) vgate + off);
                        }
                    }
                }
            }
        }
#endif

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            case GGML_GLU_OP_SWIGLU_CLAMP:
                                result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block, bool has_fusion = false>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    // fuse gate, bias, scales, and glu_op into the up projection
    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    const float * x_scale    = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;

    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        vgate      = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            x_scale    = (const float *) fusion.x_scale;
            gate_scale = (const float *) fusion.gate_scale;
        }
    }

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = shared_expert ? 0 : ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
                }
            }
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            const uint32_t bias_idx = channel_x*stride_channel_dst + row0 + threadIdx.x;

            if constexpr (type == GGML_TYPE_NVFP4) {
                if (x_scale) {
                    result *= x_scale[channel_x];
                }
            }
            if (x_bias) {
                result += x_bias[bias_idx];
            }
            if (use_gate) {
                float gate_value = tmp_gate[threadIdx.x];
                if constexpr (type == GGML_TYPE_NVFP4) {
                    if (gate_scale) {
                        gate_value *= gate_scale[channel_x];
                    }
                }
                if (gate_bias) {
                    gate_value += gate_bias[bias_idx];
                }
                switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    case GGML_GLU_OP_SWIGLU_CLAMP:
                        result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                        break;
                    default:
                        result = result * gate_value;
                        break;
                }
            }
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, tmp_gate, vgate, x_bias, gate_bias, active_glu, glu_limit, x_scale, gate_scale);
    } else if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(x_scale, gate_scale);
    }
}

#if defined(GGML_USE_HIP)
struct mmvq_nvfp4_q8_1_lds {
    int   qs[8];
    float d;
};

static __device__ __constant__ float k_ue4m3_lut[256] = {
    0.0f, 0.0009765625f, 0.001953125f, 0.0029296875f, 0.00390625f, 0.0048828125f, 0.005859375f, 0.0068359375f,
    0.0078125f, 0.0087890625f, 0.009765625f, 0.0107421875f, 0.01171875f, 0.0126953125f, 0.013671875f, 0.0146484375f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 0.0f,
    0.0f, 0.0009765625f, 0.001953125f, 0.0029296875f, 0.00390625f, 0.0048828125f, 0.005859375f, 0.0068359375f,
    0.0078125f, 0.0087890625f, 0.009765625f, 0.0107421875f, 0.01171875f, 0.0126953125f, 0.013671875f, 0.0146484375f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f
};

// GCN/CDNA NVFP4 mat-vec (ncols_dst == 1, or multi-token MUL_MAT_ID with blockIdx.z = token):
// caches q8_1 y and ue4m3 LUT in LDS, maps LPR lanes to NR rows.
template <int LPR, int NW, int NR, int MIN_BLOCKS = 1>
__launch_bounds__(64 * NW, MIN_BLOCKS)
static __global__ void mul_mat_vec_q_nvfp4_gcn(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr,
        const float * x_scale_ptr, const float * x_bias_ptr, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint3 channel_ratio,
        const uint3 sample_ratio, const uint32_t stride_sample_x, const uint32_t stride_sample_y,
        const uint32_t stride_sample_dst, const uint32_t ids_stride) {
    const block_nvfp4 * GGML_CUDA_RESTRICT vx  = (const block_nvfp4 *) vx_ptr;
    const block_q8_1  * GGML_CUDA_RESTRICT vy  = (const block_q8_1  *) vy_ptr;
    const int32_t     * GGML_CUDA_RESTRICT ids = ids_ptr;
    float             * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int n_threads = 64 * NW;
    constexpr int sub_per_wave = 64 / LPR;
    constexpr int rows_per_block = NW * sub_per_wave * NR;
    constexpr int blocks_per_iter = LPR / 2;

    const int tid = threadIdx.y * 64 + threadIdx.x;
    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;

    const uint32_t channel_x = ids ? ids[channel_dst + sample_dst * ids_stride] : fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y = ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    const uint32_t sample_x  = ids ? 0u : fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y  = sample_dst;

    const int blocks_per_row_x = ncols_x / QK_NVFP4;
    const int n_y_blocks = 2 * blocks_per_row_x;

    extern __shared__ char smem[];
    float * ue4m3_lut = (float *) smem;
    mmvq_nvfp4_q8_1_lds * ys = (mmvq_nvfp4_q8_1_lds *) (ue4m3_lut + 256);

    for (int i = tid; i < 256; i += n_threads) {
        ue4m3_lut[i] = k_ue4m3_lut[i];
    }

    const block_q8_1 * y_base = vy + (size_t) sample_y * stride_sample_y + (size_t) channel_y * stride_channel_y;
    for (int b = tid; b < n_y_blocks; b += n_threads) {
        const block_q8_1 yb = y_base[b];
        ys[b].d = __low2float(yb.ds);
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            ys[b].qs[q] = get_int_b4(yb.qs, q);
        }
    }
    __syncthreads();

    const int sub_id   = threadIdx.x / LPR;
    const int sub_lane = threadIdx.x % LPR;
    const int half_blk = sub_lane & 1;
    const int q_off    = half_blk * 4;
    const int d_off    = half_blk * 2;

    const int row_base = blockIdx.x * rows_per_block + (threadIdx.y * sub_per_wave + sub_id) * NR;
    const size_t x_base_off = (size_t) sample_x * stride_sample_x + (size_t) channel_x * stride_channel_x;

    const block_nvfp4 * x_row[NR];
    float sum[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const uint32_t row = min((uint32_t) (row_base + r), nrows_x - 1);
        x_row[r] = vx + x_base_off + (size_t) row * stride_row_x;
        sum[r] = 0.0f;
    }

    for (int kbx = sub_lane >> 1; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = (kbx << 1) + half_blk;
        const int * yq = ys[kby].qs;
        const float yd = ys[kby].d;

        int q0[NR], q1[NR], q2[NR], q3[NR];
        uint8_t d0[NR], d1[NR];
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const block_nvfp4 * bq4 = x_row[r] + kbx;
            q0[r] = get_int_b4(bq4->qs, q_off + 0);
            q1[r] = get_int_b4(bq4->qs, q_off + 1);
            q2[r] = get_int_b4(bq4->qs, q_off + 2);
            q3[r] = get_int_b4(bq4->qs, q_off + 3);
            d0[r] = bq4->d[d_off + 0];
            d1[r] = bq4->d[d_off + 1];
        }

        const int y0 = yq[0], y1 = yq[1], y2 = yq[2], y3 = yq[3];
        int sumi0[NR];
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const int2 v0 = get_int_from_table_16(q0[r], kvalues_mxfp4);
            const int2 v1 = get_int_from_table_16(q1[r], kvalues_mxfp4);
            int s = ggml_cuda_dp4a(v0.x, y0, 0);
            s = ggml_cuda_dp4a(v0.y, y2, s);
            s = ggml_cuda_dp4a(v1.x, y1, s);
            sumi0[r] = ggml_cuda_dp4a(v1.y, y3, s);
        }

        const int y4 = yq[4], y5 = yq[5], y6 = yq[6], y7 = yq[7];
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const int2 v2 = get_int_from_table_16(q2[r], kvalues_mxfp4);
            const int2 v3 = get_int_from_table_16(q3[r], kvalues_mxfp4);
            int sumi1 = ggml_cuda_dp4a(v2.x, y4, 0);
            sumi1 = ggml_cuda_dp4a(v2.y, y6, sumi1);
            sumi1 = ggml_cuda_dp4a(v3.x, y5, sumi1);
            sumi1 = ggml_cuda_dp4a(v3.y, y7, sumi1);

            sum[r] += yd * (ue4m3_lut[d0[r]] * float(sumi0[r]) + ue4m3_lut[d1[r]] * float(sumi1));
        }
    }

    const float x_scale = x_scale_ptr ? x_scale_ptr[ids ? channel_x : 0] : 1.0f;
    const float * x_bias = x_bias_ptr ? x_bias_ptr + (ids ? (size_t) channel_x * stride_channel_dst : (size_t) sample_dst * stride_sample_dst + (size_t) channel_dst * stride_channel_dst) : nullptr;
    float * dst_row = dst + (size_t) sample_dst * stride_sample_dst + (size_t) channel_dst * stride_channel_dst;

    #pragma unroll
    for (int r = 0; r < NR; ++r) {
        const float s = warp_reduce_sum<LPR>(sum[r]);
        const uint32_t row = (uint32_t) (row_base + r);
        if (sub_lane == 0 && row < nrows_x) {
            float out = s * x_scale;
            if (x_bias) {
                out += x_bias[row];
            }
            dst_row[row] = out;
        }
    }
}

template <int LPR, int NR, int M>
static __device__ __forceinline__ void nvfp4_grouped_k_loop_and_store(
        const block_nvfp4 * __restrict__ vx,
        const size_t x_base_off,
        const uint32_t stride_row_x,
        const int4 * __restrict__ s_y_qs,
        const float * __restrict__ s_y_d,
        const float * __restrict__ ue4m3_lut,
        const uint8_t * s_pairs,
        const uint3 nchannels_dst_fd,
        const uint32_t nchannels_dst,
        const uint32_t stride_col_dst,
        const uint32_t stride_channel_dst,
        const int blocks_per_row_x,
        const int n_y_blocks,
        const uint32_t nrows_x,
        const int row_base,
        const int sub_lane,
        const float x_scale,
        const float * __restrict__ x_bias,
        float * __restrict__ dst) {
    constexpr int blocks_per_iter = LPR / 2;
    const int half_blk = sub_lane & 1;
    const int q_off    = half_blk * 4;
    const int d_off    = half_blk * 2;

    const block_nvfp4 * x_row[NR];
    float sum[M][NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const uint32_t row = min((uint32_t) (row_base + r), nrows_x - 1);
        x_row[r] = vx + x_base_off + (size_t) row * stride_row_x;
#pragma unroll
        for (int c = 0; c < M; ++c) {
            sum[c][r] = 0.0f;
        }
    }

    for (int kbx = sub_lane >> 1; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = (kbx << 1) + half_blk;

        int q0[NR], q1[NR], q2[NR], q3[NR];
        uint16_t d01[NR];
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const block_nvfp4 * bq4 = x_row[r] + kbx;
            q0[r]  = get_int_b4(bq4->qs, q_off + 0);
            q1[r]  = get_int_b4(bq4->qs, q_off + 1);
            q2[r]  = get_int_b4(bq4->qs, q_off + 2);
            q3[r]  = get_int_b4(bq4->qs, q_off + 3);
            d01[r] = *(const uint16_t *) (bq4->d + d_off);
        }

#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const int2 v0 = get_int_from_table_16(q0[r], kvalues_mxfp4);
            const int2 v1 = get_int_from_table_16(q1[r], kvalues_mxfp4);
            int sumi0[M];
#pragma unroll
            for (int c = 0; c < M; ++c) {
                const int4 y03 = s_y_qs[2 * (c * n_y_blocks + kby) + 0];
                int s = ggml_cuda_dp4a(v0.x, y03.x, 0);
                s = ggml_cuda_dp4a(v0.y, y03.z, s);
                s = ggml_cuda_dp4a(v1.x, y03.y, s);
                sumi0[c] = ggml_cuda_dp4a(v1.y, y03.w, s);
            }

            const int2 v2 = get_int_from_table_16(q2[r], kvalues_mxfp4);
            const int2 v3 = get_int_from_table_16(q3[r], kvalues_mxfp4);
            const float sc0 = ue4m3_lut[d01[r] & 0xffu];
            const float sc1 = ue4m3_lut[d01[r] >> 8];
#pragma unroll
            for (int c = 0; c < M; ++c) {
                const int idx  = c * n_y_blocks + kby;
                const int4 y47 = s_y_qs[2 * idx + 1];
                const float yd = s_y_d[idx];
                int sumi1 = ggml_cuda_dp4a(v2.x, y47.x, 0);
                sumi1 = ggml_cuda_dp4a(v2.y, y47.z, sumi1);
                sumi1 = ggml_cuda_dp4a(v3.x, y47.y, sumi1);
                sumi1 = ggml_cuda_dp4a(v3.y, y47.w, sumi1);

                sum[c][r] += yd * (sc0 * float(sumi0[c]) + sc1 * float(sumi1));
            }
        }
    }

#pragma unroll
    for (int c = 0; c < M; ++c) {
        const int pair = s_pairs[c];
        const int t    = fastdiv(pair, nchannels_dst_fd);
        const int k    = pair - t * (int) nchannels_dst;
        float * dst_row = dst + (size_t) t * stride_col_dst + (size_t) k * stride_channel_dst;
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const float s = warp_reduce_sum<LPR>(sum[c][r]);
            const uint32_t row = (uint32_t) (row_base + r);
            if (sub_lane == 0 && row < nrows_x) {
                float out = s * x_scale;
                if (x_bias) {
                    out += x_bias[row];
                }
                dst_row[row] = out;
            }
        }
    }
}

// Grouped (expert-deduplicated) GCN/CDNA NVFP4 MoE mat-vec for multi-token MUL_MAT_ID (nchannels_dst * ncols_dst <= 64).
// Wave 0 groups ids into chunks of 1..4 occurrences per distinct expert so each expert's weights are read once.
template <int LPR, int NW, int NR, int MIN_BLOCKS = 3>
__launch_bounds__(64 * NW, MIN_BLOCKS)
static __global__ void mul_mat_vec_q_nvfp4_grouped_gcn(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr,
        const float * x_scale_ptr, const float * x_bias_ptr, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y_fd, const uint32_t nrows_x,
        const uint32_t nchannels_dst, const uint3 nchannels_dst_fd, const uint32_t n_pairs,
        const uint3 n_y_blocks_fd, const uint32_t stride_row_x, const uint32_t stride_channel_x,
        const uint32_t stride_col_y, const uint32_t stride_channel_y,
        const uint32_t stride_col_dst, const uint32_t stride_channel_dst,
        const uint32_t ids_stride) {
    const block_nvfp4 * GGML_CUDA_RESTRICT vx  = (const block_nvfp4 *) vx_ptr;
    const block_q8_1  * GGML_CUDA_RESTRICT vy  = (const block_q8_1  *) vy_ptr;
    const int32_t     * GGML_CUDA_RESTRICT ids = ids_ptr;
    float             * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int n_threads      = 64 * NW;
    constexpr int sub_per_wave   = 64 / LPR;
    constexpr int rows_per_block = NW * sub_per_wave * NR;

    __shared__ int     s_expert;
    __shared__ int     s_count;
    __shared__ uint8_t s_pairs[4];
    __shared__ const block_q8_1 * s_y_ptrs[4];

    if (threadIdx.y == 0) {
        const int lane = threadIdx.x;
        int my_e = -1;
        if (lane < (int) n_pairs) {
            const int t = fastdiv(lane, nchannels_dst_fd);
            const int k = lane - t * (int) nchannels_dst;
            my_e = ids[k + t * (int) ids_stride];
        }
        int occ = 0;
        for (int j = 0; j < (int) n_pairs; ++j) {
            const int other_e = __builtin_amdgcn_readlane(my_e, j);
            if (j < lane && other_e == my_e) {
                ++occ;
            }
        }
        // Experts repeating > 4 times split into chunks of 4; total chunks <= n_pairs.
        const bool is_chunk_head = (lane < (int) n_pairs) && ((occ & 3) == 0);
        const uint64_t head_mask = ggml_cuda_ballot(is_chunk_head);
        const int n_chunks = ggml_cuda_popc(head_mask);
        if (blockIdx.y >= (uint32_t) n_chunks) {
            if (lane == 0) {
                s_count = 0;
            }
        } else {
            const uint64_t lower_mask = (1ULL << lane) - 1ULL;
            const int rank = ggml_cuda_popc(head_mask & lower_mask);
            const uint64_t leader_mask = ggml_cuda_ballot(is_chunk_head && rank == (int) blockIdx.y);
            const int leader = __ffsll((long long) leader_mask) - 1;
            const int target_e = __builtin_amdgcn_readlane(my_e, leader);
            const int target_chunk = __builtin_amdgcn_readlane(occ >> 2, leader);
            const bool in_chunk = (lane < (int) n_pairs) && (my_e == target_e) && ((occ >> 2) == target_chunk);
            const uint64_t match_mask = ggml_cuda_ballot(in_chunk);
            const int count = ggml_cuda_popc(match_mask);
            if (in_chunk) {
                const int m_idx = ggml_cuda_popc(match_mask & lower_mask);
                s_pairs[m_idx] = (uint8_t) lane;
                const int t  = fastdiv(lane, nchannels_dst_fd);
                const int k  = lane - t * (int) nchannels_dst;
                const int cy = fastmodulo(k, nchannels_y_fd);
                s_y_ptrs[m_idx] = vy + (size_t) t * stride_col_y + (size_t) cy * stride_channel_y;
            }
            if (lane == 0) {
                s_expert = target_e;
                s_count  = count;
            }
        }
    }
    __syncthreads();

    const int m = s_count;
    if (m == 0) {
        return;
    }

    const uint32_t channel_x = (uint32_t) s_expert;
    const int tid = threadIdx.y * 64 + threadIdx.x;

    const int blocks_per_row_x = ncols_x / QK_NVFP4;
    const int n_y_blocks       = 2 * blocks_per_row_x;
    const int m_cap            = n_pairs < 4u ? (int) n_pairs : 4;

    extern __shared__ char smem[];
    int4  * s_y_qs    = (int4  *) smem;
    float * ue4m3_lut = (float *) (s_y_qs + (size_t) m_cap * n_y_blocks * 2);
    float * s_y_d     = ue4m3_lut + 256;

    for (int i = tid; i < 256; i += n_threads) {
        ue4m3_lut[i] = k_ue4m3_lut[i];
    }

    const int total_y = m * n_y_blocks;
    for (int idx = tid; idx < total_y; idx += n_threads) {
        const int c = m == 1 ? 0 : fastdiv(idx, n_y_blocks_fd);
        const int b = idx - c * n_y_blocks;
        const block_q8_1 yb = s_y_ptrs[c][b];
        s_y_d [idx]         = __low2float(yb.ds);
        s_y_qs[2 * idx + 0] = make_int4(get_int_b4(yb.qs, 0), get_int_b4(yb.qs, 1), get_int_b4(yb.qs, 2), get_int_b4(yb.qs, 3));
        s_y_qs[2 * idx + 1] = make_int4(get_int_b4(yb.qs, 4), get_int_b4(yb.qs, 5), get_int_b4(yb.qs, 6), get_int_b4(yb.qs, 7));
    }
    __syncthreads();

    const int sub_id   = threadIdx.x / LPR;
    const int sub_lane = threadIdx.x % LPR;
    const size_t x_base_off = (size_t) channel_x * stride_channel_x;
    const float   x_scale = x_scale_ptr ? x_scale_ptr[channel_x] : 1.0f;
    const float * x_bias  = x_bias_ptr  ? x_bias_ptr + (size_t) channel_x * stride_channel_dst : nullptr;
    const int row_base = blockIdx.x * rows_per_block + (threadIdx.y * sub_per_wave + sub_id) * NR;

    if (m == 1) {
        nvfp4_grouped_k_loop_and_store<LPR, NR, 1>(
            vx, x_base_off, stride_row_x, s_y_qs, s_y_d, ue4m3_lut, s_pairs,
            nchannels_dst_fd, nchannels_dst, stride_col_dst, stride_channel_dst,
            blocks_per_row_x, n_y_blocks, nrows_x, row_base, sub_lane, x_scale, x_bias, dst);
    } else if (m == 2) {
        nvfp4_grouped_k_loop_and_store<LPR, NR, 2>(
            vx, x_base_off, stride_row_x, s_y_qs, s_y_d, ue4m3_lut, s_pairs,
            nchannels_dst_fd, nchannels_dst, stride_col_dst, stride_channel_dst,
            blocks_per_row_x, n_y_blocks, nrows_x, row_base, sub_lane, x_scale, x_bias, dst);
    } else if (m == 3) {
        nvfp4_grouped_k_loop_and_store<LPR, NR, 3>(
            vx, x_base_off, stride_row_x, s_y_qs, s_y_d, ue4m3_lut, s_pairs,
            nchannels_dst_fd, nchannels_dst, stride_col_dst, stride_channel_dst,
            blocks_per_row_x, n_y_blocks, nrows_x, row_base, sub_lane, x_scale, x_bias, dst);
    } else {
        nvfp4_grouped_k_loop_and_store<LPR, NR, 4>(
            vx, x_base_off, stride_row_x, s_y_qs, s_y_d, ue4m3_lut, s_pairs,
            nchannels_dst_fd, nchannels_dst, stride_col_dst, stride_channel_dst,
            blocks_per_row_x, n_y_blocks, nrows_x, row_base, sub_lane, x_scale, x_bias, dst);
    }
}

// Wave-0 expert deduplication for grouped multi-token MUL_MAT_ID (n_pairs <= 64).
static __device__ __forceinline__ void mmvq_grouped_dedupe_wave0(
        const int32_t * __restrict__ ids,
        const block_q8_1 * __restrict__ vy,
        const uint3 nchannels_y_fd,
        const uint32_t nchannels_dst,
        const uint3 nchannels_dst_fd,
        const uint32_t n_pairs,
        const uint32_t stride_col_y,
        const uint32_t stride_channel_y,
        const uint32_t ids_stride,
        int & s_expert,
        int & s_count,
        uint8_t (&s_pairs)[4],
        const block_q8_1 * (&s_y_ptrs)[4]) {
    if (threadIdx.y != 0) {
        return;
    }
    const int lane = threadIdx.x;
    int my_e = -1;
    if (lane < (int) n_pairs) {
        const int t = fastdiv(lane, nchannels_dst_fd);
        const int k = lane - t * (int) nchannels_dst;
        my_e = ids[k + t * (int) ids_stride];
    }
    int occ = 0;
    for (int j = 0; j < (int) n_pairs; ++j) {
        const int other_e = __builtin_amdgcn_readlane(my_e, j);
        if (j < lane && other_e == my_e) {
            ++occ;
        }
    }
    const bool is_chunk_head = (lane < (int) n_pairs) && ((occ & 3) == 0);
    const uint64_t head_mask = ggml_cuda_ballot(is_chunk_head);
    const int n_chunks = ggml_cuda_popc(head_mask);
    if (blockIdx.y >= (uint32_t) n_chunks) {
        if (lane == 0) {
            s_count = 0;
        }
        return;
    }
    const uint64_t lower_mask = (1ULL << lane) - 1ULL;
    const int rank = ggml_cuda_popc(head_mask & lower_mask);
    const uint64_t leader_mask = ggml_cuda_ballot(is_chunk_head && rank == (int) blockIdx.y);
    const int leader = __ffsll((long long) leader_mask) - 1;
    const int target_e = __builtin_amdgcn_readlane(my_e, leader);
    const int target_chunk = __builtin_amdgcn_readlane(occ >> 2, leader);
    const bool in_chunk = (lane < (int) n_pairs) && (my_e == target_e) && ((occ >> 2) == target_chunk);
    const uint64_t match_mask = ggml_cuda_ballot(in_chunk);
    const int count = ggml_cuda_popc(match_mask);
    if (in_chunk) {
        const int m_idx = ggml_cuda_popc(match_mask & lower_mask);
        s_pairs[m_idx] = (uint8_t) lane;
        const int t  = fastdiv(lane, nchannels_dst_fd);
        const int k  = lane - t * (int) nchannels_dst;
        const int cy = fastmodulo(k, nchannels_y_fd);
        s_y_ptrs[m_idx] = vy + (size_t) t * stride_col_y + (size_t) cy * stride_channel_y;
    }
    if (lane == 0) {
        s_expert = target_e;
        s_count  = count;
    }
}

// Branchless 4-byte aligned load of a 22-byte block_q5_0 (requires 4-byte aligned base and even block counts/strides).
static __device__ __forceinline__ void load_q5_0_block_gcn(
        const block_q5_0 * __restrict__ b, const bool odd_blk, float & d, uint32_t & qh, uint32_t (&vl)[4]) {
    typedef uint32_t u32x4_u __attribute__((ext_vector_type(4), aligned(4)));
    typedef uint32_t u32x2_u __attribute__((ext_vector_type(2), aligned(4)));
    const uint32_t * p = (const uint32_t *) ((const char *) b - (odd_blk ? 2 : 0));
    const u32x4_u u0 = *(const u32x4_u *) (p + 0);
    const u32x2_u u1 = *(const u32x2_u *) (p + 4);
    const uint16_t dh = (uint16_t) (odd_blk ? (u0.x >> 16) : u0.x);
    d = __half2float(*(const half *) &dh);
    qh    = odd_blk ? u0.y : ((u0.x >> 16) | (u0.y << 16));
    vl[0] = odd_blk ? u0.z : ((u0.y >> 16) | (u0.z << 16));
    vl[1] = odd_blk ? u0.w : ((u0.z >> 16) | (u0.w << 16));
    vl[2] = odd_blk ? u1.x : ((u0.w >> 16) | (u1.x << 16));
    vl[3] = odd_blk ? u1.y : ((u1.x >> 16) | (u1.y << 16));
}

template <int LPR, int NR, int M>
static __device__ __forceinline__ void q5_0_grouped_k_loop_and_store(
        const block_q5_0 * __restrict__ vx,
        const size_t x_base_off,
        const uint32_t stride_row_x,
        const int4 * __restrict__ s_y_qs,
        const float2 * __restrict__ s_y_ds,
        const uint8_t * s_pairs,
        const uint3 nchannels_dst_fd,
        const uint32_t nchannels_dst,
        const uint32_t stride_col_dst,
        const uint32_t stride_channel_dst,
        const int blocks_per_row_x,
        const uint32_t nrows_x,
        const int row_base,
        const int sub_lane,
        const float * __restrict__ x_bias,
        float * __restrict__ dst) {
    const bool odd_blk = (sub_lane & 1) != 0;
    const block_q5_0 * x_row[NR];
    float sum[M][NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const uint32_t row = min((uint32_t) (row_base + r), nrows_x - 1);
        x_row[r] = vx + x_base_off + (size_t) row * stride_row_x;
#pragma unroll
        for (int c = 0; c < M; ++c) {
            sum[c][r] = 0.0f;
        }
    }

    for (int kb = sub_lane; kb < blocks_per_row_x; kb += LPR) {
        float    d5[NR];
        uint32_t qh[NR];
        uint32_t vl[NR][4];
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            load_q5_0_block_gcn(x_row[r] + kb, odd_blk, d5[r], qh[r], vl[r]);
        }

        int vi0[NR][4];
#pragma unroll
        for (int r = 0; r < NR; ++r) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t h = (qh[r] >> (4 * i)) & 0xFu;
                vi0[r][i] = (int) ((vl[r][i] & 0x0F0F0F0Fu) | ((h * 0x02040810u) & 0x10101010u));
            }
        }
#pragma unroll
        for (int c = 0; c < M; ++c) {
            const int idx = c * blocks_per_row_x + kb;
            const int4   y03 = s_y_qs[2 * idx + 0];
            const float2 yds = s_y_ds[idx];
#pragma unroll
            for (int r = 0; r < NR; ++r) {
                int s = ggml_cuda_dp4a(vi0[r][0], y03.x, 0);
                s = ggml_cuda_dp4a(vi0[r][1], y03.y, s);
                s = ggml_cuda_dp4a(vi0[r][2], y03.z, s);
                s = ggml_cuda_dp4a(vi0[r][3], y03.w, s);
                sum[c][r] += d5[r] * (float(s) * yds.x + yds.y);
            }
        }

        int vi1[NR][4];
#pragma unroll
        for (int r = 0; r < NR; ++r) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t h = (qh[r] >> (16 + 4 * i)) & 0xFu;
                vi1[r][i] = (int) (((vl[r][i] >> 4) & 0x0F0F0F0Fu) | ((h * 0x02040810u) & 0x10101010u));
            }
        }
#pragma unroll
        for (int c = 0; c < M; ++c) {
            const int idx = c * blocks_per_row_x + kb;
            const int4  y47 = s_y_qs[2 * idx + 1];
            const float yd  = s_y_ds[idx].x;
#pragma unroll
            for (int r = 0; r < NR; ++r) {
                int s = ggml_cuda_dp4a(vi1[r][0], y47.x, 0);
                s = ggml_cuda_dp4a(vi1[r][1], y47.y, s);
                s = ggml_cuda_dp4a(vi1[r][2], y47.z, s);
                s = ggml_cuda_dp4a(vi1[r][3], y47.w, s);
                sum[c][r] += (d5[r] * yd) * float(s);
            }
        }
    }

#pragma unroll
    for (int c = 0; c < M; ++c) {
        const int pair = s_pairs[c];
        const int t    = fastdiv(pair, nchannels_dst_fd);
        const int k    = pair - t * (int) nchannels_dst;
        float * dst_row = dst + (size_t) t * stride_col_dst + (size_t) k * stride_channel_dst;
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const float s = warp_reduce_sum<LPR>(sum[c][r]);
            const uint32_t row = (uint32_t) (row_base + r);
            if (sub_lane == 0 && row < nrows_x) {
                float out = s;
                if (x_bias) {
                    out += x_bias[row];
                }
                dst_row[row] = out;
            }
        }
    }
}

template <int LPR, int NW, int NR, int MIN_BLOCKS = 3>
__launch_bounds__(64 * NW, MIN_BLOCKS)
static __global__ void mul_mat_vec_q5_0_grouped_gcn(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr,
        const float * x_bias_ptr, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y_fd, const uint32_t nrows_x,
        const uint32_t nchannels_dst, const uint3 nchannels_dst_fd, const uint32_t n_pairs,
        const uint3 n_y_blocks_fd, const uint32_t stride_row_x, const uint32_t stride_channel_x,
        const uint32_t stride_col_y, const uint32_t stride_channel_y,
        const uint32_t stride_col_dst, const uint32_t stride_channel_dst,
        const uint32_t ids_stride) {
    const block_q5_0 * GGML_CUDA_RESTRICT vx  = (const block_q5_0 *) vx_ptr;
    const block_q8_1 * GGML_CUDA_RESTRICT vy  = (const block_q8_1 *) vy_ptr;
    const int32_t    * GGML_CUDA_RESTRICT ids = ids_ptr;
    float            * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int n_threads      = 64 * NW;
    constexpr int sub_per_wave   = 64 / LPR;
    constexpr int rows_per_block = NW * sub_per_wave * NR;

    __shared__ int     s_expert;
    __shared__ int     s_count;
    __shared__ uint8_t s_pairs[4];
    __shared__ const block_q8_1 * s_y_ptrs[4];

    mmvq_grouped_dedupe_wave0(
        ids, vy, nchannels_y_fd, nchannels_dst, nchannels_dst_fd, n_pairs,
        stride_col_y, stride_channel_y, ids_stride,
        s_expert, s_count, s_pairs, s_y_ptrs);
    __syncthreads();

    const int m = s_count;
    if (m == 0) {
        return;
    }

    const uint32_t channel_x = (uint32_t) s_expert;
    const int tid = threadIdx.y * 64 + threadIdx.x;
    const int blocks_per_row_x = ncols_x / QK5_0;
    const int m_cap = n_pairs < 4u ? (int) n_pairs : 4;

    extern __shared__ char smem[];
    int4   * s_y_qs = (int4   *) smem;
    float2 * s_y_ds = (float2 *) (s_y_qs + (size_t) m_cap * blocks_per_row_x * 2);

    const int total_y = m * blocks_per_row_x;
    for (int idx = tid; idx < total_y; idx += n_threads) {
        const int c = m == 1 ? 0 : fastdiv(idx, n_y_blocks_fd);
        const int b = idx - c * blocks_per_row_x;
        const block_q8_1 yb = s_y_ptrs[c][b];
        const float2 ds = __half22float2(yb.ds);
        s_y_ds[idx]         = make_float2(ds.x, -16.0f * ds.y);
        s_y_qs[2 * idx + 0] = make_int4(get_int_b4(yb.qs, 0), get_int_b4(yb.qs, 1), get_int_b4(yb.qs, 2), get_int_b4(yb.qs, 3));
        s_y_qs[2 * idx + 1] = make_int4(get_int_b4(yb.qs, 4), get_int_b4(yb.qs, 5), get_int_b4(yb.qs, 6), get_int_b4(yb.qs, 7));
    }
    __syncthreads();

    const int sub_id   = threadIdx.x / LPR;
    const int sub_lane = threadIdx.x % LPR;
    const size_t x_base_off = (size_t) channel_x * stride_channel_x;
    const float * x_bias    = x_bias_ptr ? x_bias_ptr + (size_t) channel_x * stride_channel_dst : nullptr;
    const int row_base      = blockIdx.x * rows_per_block + (threadIdx.y * sub_per_wave + sub_id) * NR;

    if (m == 1) {
        q5_0_grouped_k_loop_and_store<LPR, NR, 1>(
            vx, x_base_off, stride_row_x, s_y_qs, s_y_ds, s_pairs,
            nchannels_dst_fd, nchannels_dst, stride_col_dst, stride_channel_dst,
            blocks_per_row_x, nrows_x, row_base, sub_lane, x_bias, dst);
    } else if (m == 2) {
        q5_0_grouped_k_loop_and_store<LPR, NR, 2>(
            vx, x_base_off, stride_row_x, s_y_qs, s_y_ds, s_pairs,
            nchannels_dst_fd, nchannels_dst, stride_col_dst, stride_channel_dst,
            blocks_per_row_x, nrows_x, row_base, sub_lane, x_bias, dst);
    } else if (m == 3) {
        q5_0_grouped_k_loop_and_store<LPR, NR, 3>(
            vx, x_base_off, stride_row_x, s_y_qs, s_y_ds, s_pairs,
            nchannels_dst_fd, nchannels_dst, stride_col_dst, stride_channel_dst,
            blocks_per_row_x, nrows_x, row_base, sub_lane, x_bias, dst);
    } else {
        q5_0_grouped_k_loop_and_store<LPR, NR, 4>(
            vx, x_base_off, stride_row_x, s_y_qs, s_y_ds, s_pairs,
            nchannels_dst_fd, nchannels_dst, stride_col_dst, stride_channel_dst,
            blocks_per_row_x, nrows_x, row_base, sub_lane, x_bias, dst);
    }
}

template <int LPR, int NR, int M>
static __device__ __forceinline__ void q4_K_grouped_k_loop_and_store(
        const block_q4_K * __restrict__ vx,
        const size_t x_base_off,
        const uint32_t stride_row_x,
        const int4 * __restrict__ s_y_qs,
        const float2 * __restrict__ s_y_ds,
        const uint8_t * s_pairs,
        const uint3 nchannels_dst_fd,
        const uint32_t nchannels_dst,
        const uint32_t stride_col_dst,
        const uint32_t stride_channel_dst,
        const int blocks_per_row_x,
        const int n_y_blocks,
        const uint32_t nrows_x,
        const int row_base,
        const int sub_lane,
        const float * __restrict__ x_bias,
        float * __restrict__ dst) {
    typedef uint32_t u32x4_u __attribute__((ext_vector_type(4), aligned(4)));
    static_assert(LPR >= 4 && (LPR % 4) == 0, "LPR must be a multiple of 4");
    constexpr int blocks_per_iter = LPR / 4;
    const int g  = sub_lane & 3;
    const int jm = g & 1;
    const uint32_t hi = (uint32_t) -(int32_t) (g >= 2);

    const block_q4_K * x_row[NR];
    float sum[M][NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const uint32_t row = min((uint32_t) (row_base + r), nrows_x - 1);
        x_row[r] = vx + x_base_off + (size_t) row * stride_row_x;
#pragma unroll
        for (int c = 0; c < M; ++c) {
            sum[c][r] = 0.0f;
        }
    }

    for (int kbx = sub_lane >> 2; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby0 = (kbx << 3) + (g << 1);

        u32x4_u hdr[NR], q03[NR], q47[NR];
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const block_q4_K * bq4 = x_row[r] + kbx;
            hdr[r] = *(const u32x4_u *) bq4;
            q03[r] = *(const u32x4_u *) (bq4->qs + 32 * g + 0);
            q47[r] = *(const u32x4_u *) (bq4->qs + 32 * g + 16);
        }

#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const uint32_t dm_bits = hdr[r].x;
            const float2 dm = __half22float2(*(const half2 *) &dm_bits);
            const uint32_t s0 = (hdr[r].y >> (16 * jm)) & 0xffffu;
            const uint32_t s2 = (hdr[r].z >> (16 * jm)) & 0xffffu;
            const uint32_t s4 = (hdr[r].w >> (16 * jm)) & 0xffffu;
            const uint32_t aux0 = (((s0 & 0x3f3fu) & ~hi) | ((((s4 >> 0) & 0x0f0fu) | ((s0 & 0xc0c0u) >> 2)) & hi));
            const uint32_t aux1 = (((s2 & 0x3f3fu) & ~hi) | ((((s4 >> 4) & 0x0f0fu) | ((s2 & 0xc0c0u) >> 2)) & hi));
            const float d_sc0 = dm.x * float(aux0 & 0xffu);
            const float d_sc1 = dm.x * float((aux0 >> 8) & 0xffu);
            const float d_m0  = dm.y * float(aux1 & 0xffu);
            const float d_m1  = dm.y * float((aux1 >> 8) & 0xffu);

            const int v0 = (int) (q03[r].x & 0x0F0F0F0Fu);
            const int v1 = (int) (q03[r].y & 0x0F0F0F0Fu);
            const int v2 = (int) (q03[r].z & 0x0F0F0F0Fu);
            const int v3 = (int) (q03[r].w & 0x0F0F0F0Fu);
            const int v4 = (int) (q47[r].x & 0x0F0F0F0Fu);
            const int v5 = (int) (q47[r].y & 0x0F0F0F0Fu);
            const int v6 = (int) (q47[r].z & 0x0F0F0F0Fu);
            const int v7 = (int) (q47[r].w & 0x0F0F0F0Fu);
#pragma unroll
            for (int c = 0; c < M; ++c) {
                const int idx0 = c * n_y_blocks + kby0;
                const int4   y03 = s_y_qs[2 * idx0 + 0];
                const int4   y47 = s_y_qs[2 * idx0 + 1];
                const float2 ds0 = s_y_ds[idx0];
                int s = ggml_cuda_dp4a(v0, y03.x, 0);
                s = ggml_cuda_dp4a(v1, y03.y, s);
                s = ggml_cuda_dp4a(v2, y03.z, s);
                s = ggml_cuda_dp4a(v3, y03.w, s);
                s = ggml_cuda_dp4a(v4, y47.x, s);
                s = ggml_cuda_dp4a(v5, y47.y, s);
                s = ggml_cuda_dp4a(v6, y47.z, s);
                s = ggml_cuda_dp4a(v7, y47.w, s);
                sum[c][r] += (d_sc0 * ds0.x) * float(s) - d_m0 * ds0.y;
            }

            const int u0 = (int) ((q03[r].x >> 4) & 0x0F0F0F0Fu);
            const int u1 = (int) ((q03[r].y >> 4) & 0x0F0F0F0Fu);
            const int u2 = (int) ((q03[r].z >> 4) & 0x0F0F0F0Fu);
            const int u3 = (int) ((q03[r].w >> 4) & 0x0F0F0F0Fu);
            const int u4 = (int) ((q47[r].x >> 4) & 0x0F0F0F0Fu);
            const int u5 = (int) ((q47[r].y >> 4) & 0x0F0F0F0Fu);
            const int u6 = (int) ((q47[r].z >> 4) & 0x0F0F0F0Fu);
            const int u7 = (int) ((q47[r].w >> 4) & 0x0F0F0F0Fu);
#pragma unroll
            for (int c = 0; c < M; ++c) {
                const int idx1 = c * n_y_blocks + kby0 + 1;
                const int4   y03 = s_y_qs[2 * idx1 + 0];
                const int4   y47 = s_y_qs[2 * idx1 + 1];
                const float2 ds1 = s_y_ds[idx1];
                int s = ggml_cuda_dp4a(u0, y03.x, 0);
                s = ggml_cuda_dp4a(u1, y03.y, s);
                s = ggml_cuda_dp4a(u2, y03.z, s);
                s = ggml_cuda_dp4a(u3, y03.w, s);
                s = ggml_cuda_dp4a(u4, y47.x, s);
                s = ggml_cuda_dp4a(u5, y47.y, s);
                s = ggml_cuda_dp4a(u6, y47.z, s);
                s = ggml_cuda_dp4a(u7, y47.w, s);
                sum[c][r] += (d_sc1 * ds1.x) * float(s) - d_m1 * ds1.y;
            }
        }
    }

#pragma unroll
    for (int c = 0; c < M; ++c) {
        const int pair = s_pairs[c];
        const int t    = fastdiv(pair, nchannels_dst_fd);
        const int k    = pair - t * (int) nchannels_dst;
        float * dst_row = dst + (size_t) t * stride_col_dst + (size_t) k * stride_channel_dst;
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const float s = warp_reduce_sum<LPR>(sum[c][r]);
            const uint32_t row = (uint32_t) (row_base + r);
            if (sub_lane == 0 && row < nrows_x) {
                float out = s;
                if (x_bias) {
                    out += x_bias[row];
                }
                dst_row[row] = out;
            }
        }
    }
}

template <int LPR, int NW, int NR, int MIN_BLOCKS = 3>
__launch_bounds__(64 * NW, MIN_BLOCKS)
static __global__ void mul_mat_vec_q4_K_grouped_gcn(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr,
        const float * x_bias_ptr, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y_fd, const uint32_t nrows_x,
        const uint32_t nchannels_dst, const uint3 nchannels_dst_fd, const uint32_t n_pairs,
        const uint3 n_y_blocks_fd, const uint32_t stride_row_x, const uint32_t stride_channel_x,
        const uint32_t stride_col_y, const uint32_t stride_channel_y,
        const uint32_t stride_col_dst, const uint32_t stride_channel_dst,
        const uint32_t ids_stride) {
    const block_q4_K * GGML_CUDA_RESTRICT vx  = (const block_q4_K *) vx_ptr;
    const block_q8_1 * GGML_CUDA_RESTRICT vy  = (const block_q8_1 *) vy_ptr;
    const int32_t    * GGML_CUDA_RESTRICT ids = ids_ptr;
    float            * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int n_threads      = 64 * NW;
    constexpr int sub_per_wave   = 64 / LPR;
    constexpr int rows_per_block = NW * sub_per_wave * NR;

    __shared__ int     s_expert;
    __shared__ int     s_count;
    __shared__ uint8_t s_pairs[4];
    __shared__ const block_q8_1 * s_y_ptrs[4];

    mmvq_grouped_dedupe_wave0(
        ids, vy, nchannels_y_fd, nchannels_dst, nchannels_dst_fd, n_pairs,
        stride_col_y, stride_channel_y, ids_stride,
        s_expert, s_count, s_pairs, s_y_ptrs);
    __syncthreads();

    const int m = s_count;
    if (m == 0) {
        return;
    }

    const uint32_t channel_x = (uint32_t) s_expert;
    const int tid = threadIdx.y * 64 + threadIdx.x;
    const int blocks_per_row_x = ncols_x / QK_K;
    const int n_y_blocks       = 8 * blocks_per_row_x;
    const int m_cap            = n_pairs < 4u ? (int) n_pairs : 4;

    extern __shared__ char smem[];
    int4   * s_y_qs = (int4   *) smem;
    float2 * s_y_ds = (float2 *) (s_y_qs + (size_t) m_cap * n_y_blocks * 2);

    const int total_y = m * n_y_blocks;
    for (int idx = tid; idx < total_y; idx += n_threads) {
        const int c = m == 1 ? 0 : fastdiv(idx, n_y_blocks_fd);
        const int b = idx - c * n_y_blocks;
        const block_q8_1 yb = s_y_ptrs[c][b];
        s_y_ds[idx]         = __half22float2(yb.ds);
        s_y_qs[2 * idx + 0] = make_int4(get_int_b4(yb.qs, 0), get_int_b4(yb.qs, 1), get_int_b4(yb.qs, 2), get_int_b4(yb.qs, 3));
        s_y_qs[2 * idx + 1] = make_int4(get_int_b4(yb.qs, 4), get_int_b4(yb.qs, 5), get_int_b4(yb.qs, 6), get_int_b4(yb.qs, 7));
    }
    __syncthreads();

    const int sub_id   = threadIdx.x / LPR;
    const int sub_lane = threadIdx.x % LPR;
    const size_t x_base_off = (size_t) channel_x * stride_channel_x;
    const float * x_bias    = x_bias_ptr ? x_bias_ptr + (size_t) channel_x * stride_channel_dst : nullptr;
    const int row_base      = blockIdx.x * rows_per_block + (threadIdx.y * sub_per_wave + sub_id) * NR;

    if (m == 1) {
        q4_K_grouped_k_loop_and_store<LPR, NR, 1>(
            vx, x_base_off, stride_row_x, s_y_qs, s_y_ds, s_pairs,
            nchannels_dst_fd, nchannels_dst, stride_col_dst, stride_channel_dst,
            blocks_per_row_x, n_y_blocks, nrows_x, row_base, sub_lane, x_bias, dst);
    } else if (m == 2) {
        q4_K_grouped_k_loop_and_store<LPR, NR, 2>(
            vx, x_base_off, stride_row_x, s_y_qs, s_y_ds, s_pairs,
            nchannels_dst_fd, nchannels_dst, stride_col_dst, stride_channel_dst,
            blocks_per_row_x, n_y_blocks, nrows_x, row_base, sub_lane, x_bias, dst);
    } else if (m == 3) {
        q4_K_grouped_k_loop_and_store<LPR, NR, 3>(
            vx, x_base_off, stride_row_x, s_y_qs, s_y_ds, s_pairs,
            nchannels_dst_fd, nchannels_dst, stride_col_dst, stride_channel_dst,
            blocks_per_row_x, n_y_blocks, nrows_x, row_base, sub_lane, x_bias, dst);
    } else {
        q4_K_grouped_k_loop_and_store<LPR, NR, 4>(
            vx, x_base_off, stride_row_x, s_y_qs, s_y_ds, s_pairs,
            nchannels_dst_fd, nchannels_dst, stride_col_dst, stride_channel_dst,
            blocks_per_row_x, n_y_blocks, nrows_x, row_base, sub_lane, x_bias, dst);
    }
}

// branchless 4-byte aligned load of a 34-byte block_q8_0 (requires base vx 4-byte aligned)
static __device__ __forceinline__ void load_q8_0_block_gcn(const block_q8_0 * __restrict__ b, float & d, int (&v)[8]) {
    typedef uint32_t u32x4_u __attribute__((ext_vector_type(4), aligned(4)));
    const uintptr_t addr = (uintptr_t) b;
    const uint32_t  mis  = (uint32_t) (addr & 2u);
    const uint32_t * p   = (const uint32_t *) (addr - mis);
    const u32x4_u u0 = *(const u32x4_u *) (p + 0);
    const u32x4_u u1 = *(const u32x4_u *) (p + 4);
    const uint16_t * q8 = (const uint16_t *) (p + 8);
    const uint32_t w8 = (uint32_t) q8[0] | (mis ? ((uint32_t) q8[1] << 16) : 0u);
    const uint16_t dh = (uint16_t) (mis ? (u0.x >> 16) : u0.x);
    d = __half2float(*(const half *) &dh);
    if (mis) {
        v[0] = (int) u0.y;
        v[1] = (int) u0.z;
        v[2] = (int) u0.w;
        v[3] = (int) u1.x;
        v[4] = (int) u1.y;
        v[5] = (int) u1.z;
        v[6] = (int) u1.w;
        v[7] = (int) w8;
    } else {
        v[0] = (int) ((u0.x >> 16) | (u0.y << 16));
        v[1] = (int) ((u0.y >> 16) | (u0.z << 16));
        v[2] = (int) ((u0.z >> 16) | (u0.w << 16));
        v[3] = (int) ((u0.w >> 16) | (u1.x << 16));
        v[4] = (int) ((u1.x >> 16) | (u1.y << 16));
        v[5] = (int) ((u1.y >> 16) | (u1.z << 16));
        v[6] = (int) ((u1.z >> 16) | (u1.w << 16));
        v[7] = (int) ((u1.w >> 16) | (w8   << 16));
    }
}

// GCN/CDNA Q8_0 mat-vec (ncols_dst 1..4): tiles q8_1 y in LDS and maps LPR lanes to NR rows.
template <int LPR, int NW, int NR, int NCOLS, int K_TILE>
__launch_bounds__(64 * NW, 2)
static __global__ void mul_mat_vec_q8_0_gcn(
        const void * vx_ptr, const void * vy_ptr, const float * x_bias_ptr, float * dst_ptr,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint3 channel_ratio, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {
    const block_q8_0 * GGML_CUDA_RESTRICT vx  = (const block_q8_0 *) vx_ptr;
    const block_q8_1 * GGML_CUDA_RESTRICT vy  = (const block_q8_1 *) vy_ptr;
    float            * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int n_threads      = 64 * NW;
    constexpr int sub_per_wave   = 64 / LPR;
    constexpr int rows_per_block = NW * sub_per_wave * NR;

    const int tid = threadIdx.y * 64 + threadIdx.x;
    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;

    const uint32_t channel_x = fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y = channel_dst;
    const uint32_t sample_x  = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y  = sample_dst;

    const int blocks_per_row_x = ncols_x / QK8_0;
    const int sub_id   = threadIdx.x / LPR;
    const int sub_lane = threadIdx.x % LPR;

    const int row_base = blockIdx.x * rows_per_block + (threadIdx.y * sub_per_wave + sub_id) * NR;
    const size_t x_base_off = (size_t) sample_x * stride_sample_x + (size_t) channel_x * stride_channel_x;

    const block_q8_0 * x_row[NR];
    float sum[NCOLS][NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const uint32_t row = min((uint32_t) (row_base + r), nrows_x - 1);
        x_row[r] = vx + x_base_off + (size_t) row * stride_row_x;
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            sum[c][r] = 0.0f;
        }
    }

    const block_q8_1 * y_base = vy + (size_t) sample_y * stride_sample_y + (size_t) channel_y * stride_channel_y;

    __shared__ int4  s_y_qs[NCOLS][K_TILE][2];
    __shared__ float s_y_d [NCOLS][K_TILE];

    for (int t = 0; t < blocks_per_row_x; t += K_TILE) {
        const int tile_len = min(K_TILE, blocks_per_row_x - t);
        if constexpr (NCOLS == 1) {
            for (int kb = tid; kb < tile_len; kb += n_threads) {
                const block_q8_1 yb = y_base[t + kb];
                s_y_d[0][kb] = __low2float(yb.ds);
                s_y_qs[0][kb][0] = make_int4(get_int_b4(yb.qs, 0), get_int_b4(yb.qs, 1), get_int_b4(yb.qs, 2), get_int_b4(yb.qs, 3));
                s_y_qs[0][kb][1] = make_int4(get_int_b4(yb.qs, 4), get_int_b4(yb.qs, 5), get_int_b4(yb.qs, 6), get_int_b4(yb.qs, 7));
            }
        } else {
            const int total_load = NCOLS * tile_len;
            for (int idx = tid; idx < total_load; idx += n_threads) {
                const int c  = idx / tile_len;
                const int kb = idx - c * tile_len;
                const block_q8_1 yb = y_base[(size_t) c * stride_col_y + t + kb];
                s_y_d[c][kb] = __low2float(yb.ds);
                s_y_qs[c][kb][0] = make_int4(get_int_b4(yb.qs, 0), get_int_b4(yb.qs, 1), get_int_b4(yb.qs, 2), get_int_b4(yb.qs, 3));
                s_y_qs[c][kb][1] = make_int4(get_int_b4(yb.qs, 4), get_int_b4(yb.qs, 5), get_int_b4(yb.qs, 6), get_int_b4(yb.qs, 7));
            }
        }
        __syncthreads();

        for (int kb = sub_lane; kb < tile_len; kb += LPR) {
            if constexpr (NCOLS == 1) {
                const int4  y0 = s_y_qs[0][kb][0];
                const int4  y1 = s_y_qs[0][kb][1];
                const float yd = s_y_d [0][kb];
#pragma unroll 1
                for (int r = 0; r < NR; ++r) {
                    float xd;
                    int xq[8];
                    load_q8_0_block_gcn(x_row[r] + t + kb, xd, xq);
                    int s = ggml_cuda_dp4a(xq[0], y0.x, 0);
                    s = ggml_cuda_dp4a(xq[1], y0.y, s);
                    s = ggml_cuda_dp4a(xq[2], y0.z, s);
                    s = ggml_cuda_dp4a(xq[3], y0.w, s);
                    s = ggml_cuda_dp4a(xq[4], y1.x, s);
                    s = ggml_cuda_dp4a(xq[5], y1.y, s);
                    s = ggml_cuda_dp4a(xq[6], y1.z, s);
                    s = ggml_cuda_dp4a(xq[7], y1.w, s);
                    sum[0][r] += (xd * yd) * float(s);
                }
            } else {
                float xd[NR];
                int xq[NR][8];
#pragma unroll
                for (int r = 0; r < NR; ++r) {
                    load_q8_0_block_gcn(x_row[r] + t + kb, xd[r], xq[r]);
                }
#pragma unroll
                for (int c = 0; c < NCOLS; ++c) {
                    const int4  y0 = s_y_qs[c][kb][0];
                    const int4  y1 = s_y_qs[c][kb][1];
                    const float yd = s_y_d [c][kb];
#pragma unroll
                    for (int r = 0; r < NR; ++r) {
                        int s = ggml_cuda_dp4a(xq[r][0], y0.x, 0);
                        s = ggml_cuda_dp4a(xq[r][1], y0.y, s);
                        s = ggml_cuda_dp4a(xq[r][2], y0.z, s);
                        s = ggml_cuda_dp4a(xq[r][3], y0.w, s);
                        s = ggml_cuda_dp4a(xq[r][4], y1.x, s);
                        s = ggml_cuda_dp4a(xq[r][5], y1.y, s);
                        s = ggml_cuda_dp4a(xq[r][6], y1.z, s);
                        s = ggml_cuda_dp4a(xq[r][7], y1.w, s);
                        sum[c][r] += (xd[r] * yd) * float(s);
                    }
                }
            }
        }

        if (t + K_TILE < blocks_per_row_x) {
            __syncthreads();
        }
    }

    const float * x_bias = x_bias_ptr ? x_bias_ptr + (size_t) sample_dst * stride_sample_dst + (size_t) channel_dst * stride_channel_dst : nullptr;
    float * dst_base = dst + (size_t) sample_dst * stride_sample_dst + (size_t) channel_dst * stride_channel_dst;

#pragma unroll
    for (int c = 0; c < NCOLS; ++c) {
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const float s = warp_reduce_sum<LPR>(sum[c][r]);
            const uint32_t row = (uint32_t) (row_base + r);
            if (sub_lane == 0 && row < nrows_x) {
                float out = s;
                if (x_bias) {
                    out += x_bias[(size_t) c * stride_col_dst + row];
                }
                dst_base[(size_t) c * stride_col_dst + row] = out;
            }
        }
    }
}

// fused inkling2_situ + quantize_q8_1 so MoE ffn_down skips the F32 situ buffer
__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
static __global__ void quantize_inkling2_situ_q8_1(
        const float * __restrict__ gate, const float * __restrict__ up, void * __restrict__ vy,
        const int64_t ne00, const int64_t sg1, const int64_t sg2, const int64_t sg3,
        const int64_t su1, const int64_t su2, const int64_t su3,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2, const float beta, const float lb) {
    const int64_t i0 = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;
    if (i0 >= ne0) {
        return;
    }

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3 * ne2.z;
    const int64_t i1 = blockIdx.y;

    const int64_t i_cont = ((i3 * ne2.z + i2) * ne1 + i1) * ne0 + i0;
    block_q8_1 * y = (block_q8_1 *) vy;
    const int64_t ib  = i_cont / QK8_1;
    const int64_t iqs = i_cont % QK8_1;

    float xi = 0.0f;
    if (i0 < ne00) {
        const float g = gate[i3 * sg3 + i2 * sg2 + i1 * sg1 + i0];
        const float u = up  [i3 * su3 + i2 * su2 + i1 * su1 + i0];
        const float a = tanhf(g * (1.0f / beta)) * beta * (1.0f / (1.0f + expf(-g)));
        const float b = lb > 0.0f ? tanhf(u * (1.0f / lb)) * lb : u;
        xi = a * b;
    }

    float amax = fabsf(xi);
    amax = warp_reduce_max<QK8_1>(amax);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    const int sum  = warp_reduce_sum<QK8_1>((int) q);

    y[ib].qs[iqs] = q;
    if (iqs == 0) {
        y[ib].ds = make_half2(d, d * sum);
    }
}
#endif

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    const int rpb = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst + (fusion.shared_up != nullptr));
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;

    if (has_fusion) {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, true>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, false>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    }
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;

    // How the K loop divides up at the baseline block width, both decisions below use these.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        const int  nwarps = calc_nwarps(type, c_ncols_dst, table_id);
        bool       use    = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    // Whether doubling nwarps pays off on the ncols_dst == 1 path, where K sets the K loop trip count.
    const auto should_halve_iters = [&] {
        if (table_id != MMVQ_PARAMETERS_GB10) {
            return false;
        }

        // Expert rows are gathered per token, so a wider block adds reduction work without reuse.
        if (has_ids) {
            return false;
        }

        const int blocks_per_iter = calc_nwarps(type, 1, table_id) * blocks_per_iter_1warp;
        const int iters           = (blocks_per_row_x + blocks_per_iter - 1) /  blocks_per_iter;
        const int iters_wide      = (blocks_per_row_x + blocks_per_iter * 2 - 1) / (blocks_per_iter * 2);

        // An odd trip count leaves half the wider block idle for its last iteration, that tail is
        // only affordable once the loop is long enough to dilute it to an eighth of the work (observation).
        const int idle = iters_wide * 2 - iters;

        return idle * 8 <= iters_wide * 2;
    };

#if defined(GGML_USE_HIP)
    if constexpr (type == GGML_TYPE_NVFP4) {
        if ((ncols_dst == 1 || (has_ids && nsamples_dst == 1)) &&
            table_id == MMVQ_PARAMETERS_GCN && warp_size == 64 &&
            fusion.gate == nullptr && fusion.shared_up == nullptr &&
            fusion.gate_bias == nullptr && fusion.gate_scale == nullptr &&
            ncols_x <= 8192) {
            const int n_y_blocks = 2 * (ncols_x / QK_NVFP4);
            if (has_ids && ncols_dst > 1 && nchannels_dst > 0 && (uint64_t) nchannels_dst * ncols_dst <= 64) {
                const uint32_t n_pairs = (uint32_t) (nchannels_dst * ncols_dst);
                const int m_cap = n_pairs < 4u ? (int) n_pairs : 4;
                const int smem  = 256 * (int) sizeof(float) + m_cap * n_y_blocks * (2 * (int) sizeof(int4) + (int) sizeof(float));
                const uint3 nchannels_dst_fd = init_fastdiv_values(nchannels_dst);
                const uint3 n_y_blocks_fd    = init_fastdiv_values(n_y_blocks);
                constexpr int LPR = 8;
                constexpr int NW  = 4;
                constexpr int NR  = 2;
                constexpr int rpb = NW * (64 / LPR) * NR;
                const dim3 grid((nrows_x + rpb - 1) / rpb, n_pairs, 1);
                const dim3 block(64, NW, 1);
                mul_mat_vec_q_nvfp4_grouped_gcn<LPR, NW, NR, 3><<<grid, block, smem, stream>>>(
                    vx, vy, ids, (const float *) fusion.x_scale, (const float *) fusion.x_bias, dst,
                    ncols_x, nchannels_y_fd, nrows_x,
                    (uint32_t) nchannels_dst, nchannels_dst_fd, n_pairs, n_y_blocks_fd,
                    stride_row_x, stride_channel_x, stride_col_y, stride_channel_y,
                    stride_col_dst, stride_channel_dst, ids_stride);
                return;
            }
            const int smem = 256 * (int) sizeof(float) + n_y_blocks * (int) sizeof(mmvq_nvfp4_q8_1_lds);
            const uint32_t z_dim    = has_ids && ncols_dst > 1 ? (uint32_t) ncols_dst     : (uint32_t) nsamples_dst;
            const uint32_t sy_step  = has_ids && ncols_dst > 1 ? (uint32_t) stride_col_y   : (uint32_t) stride_sample_y;
            const uint32_t sd_step  = has_ids && ncols_dst > 1 ? (uint32_t) stride_col_dst : (uint32_t) stride_sample_dst;
            const uint32_t id_step  = has_ids && ncols_dst > 1 ? (uint32_t) ids_stride     : 0u;
            auto launch_nvfp4 = [&](auto lpr_tag) {
                constexpr int LPR = decltype(lpr_tag)::value;
                constexpr int NW  = 4;
                constexpr int NR  = 2;
                constexpr int rpb = NW * (64 / LPR) * NR;
                const dim3 grid((nrows_x + rpb - 1) / rpb, nchannels_dst, z_dim);
                const dim3 block(64, NW, 1);
                mul_mat_vec_q_nvfp4_gcn<LPR, NW, NR, 7><<<grid, block, smem, stream>>>(
                    vx, vy, ids, (const float *) fusion.x_scale, (const float *) fusion.x_bias, dst,
                    ncols_x, nchannels_y_fd, nrows_x, stride_row_x, stride_channel_x, stride_channel_y,
                    stride_channel_dst, channel_ratio_fd, sample_ratio_fd, stride_sample_x, sy_step,
                    sd_step, id_step);
            };
            if (ncols_x >= 2048) {
                launch_nvfp4(std::integral_constant<int, 16>{});
            } else {
                launch_nvfp4(std::integral_constant<int, 8>{});
            }
            return;
        }
    }
    if constexpr (type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q4_K) {
        const bool q5_aligned = (type != GGML_TYPE_Q5_0) ||
            (((blocks_per_row_x & 1) == 0) && ((stride_row_x & 1) == 0) && ((stride_channel_x & 1) == 0));
        if (has_ids && ncols_dst > 1 && nsamples_dst == 1 &&
            table_id == MMVQ_PARAMETERS_GCN && warp_size == 64 &&
            fusion.gate == nullptr && fusion.shared_up == nullptr &&
            fusion.gate_bias == nullptr && fusion.gate_scale == nullptr &&
            fusion.x_scale == nullptr &&
            ncols_x <= 8192 && ((uintptr_t) vx & 3u) == 0u && q5_aligned &&
            nchannels_dst > 0 && (uint64_t) nchannels_dst * ncols_dst <= 64) {
            const int n_y_blocks = ncols_x / QK8_1;
            const uint32_t n_pairs = (uint32_t) (nchannels_dst * ncols_dst);
            const int m_cap = n_pairs < 4u ? (int) n_pairs : 4;
            const int smem  = m_cap * n_y_blocks * (2 * (int) sizeof(int4) + (int) sizeof(float2));
            const uint3 nchannels_dst_fd = init_fastdiv_values(nchannels_dst);
            const uint3 n_y_blocks_fd    = init_fastdiv_values(n_y_blocks);
            constexpr int NW  = 4;
            const dim3 block(64, NW, 1);
            if constexpr (type == GGML_TYPE_Q5_0) {
                constexpr int LPR = 8;
                constexpr int NR  = 2;
                constexpr int rpb = NW * (64 / LPR) * NR;
                const dim3 grid((nrows_x + rpb - 1) / rpb, n_pairs, 1);
                mul_mat_vec_q5_0_grouped_gcn<LPR, NW, NR, 4><<<grid, block, smem, stream>>>(
                    vx, vy, ids, (const float *) fusion.x_bias, dst,
                    ncols_x, nchannels_y_fd, nrows_x,
                    (uint32_t) nchannels_dst, nchannels_dst_fd, n_pairs, n_y_blocks_fd,
                    stride_row_x, stride_channel_x, stride_col_y, stride_channel_y,
                    stride_col_dst, stride_channel_dst, ids_stride);
            } else {
                constexpr int LPR = 4;
                constexpr int NR  = 1;
                constexpr int rpb = NW * (64 / LPR) * NR;
                const dim3 grid((nrows_x + rpb - 1) / rpb, n_pairs, 1);
                mul_mat_vec_q4_K_grouped_gcn<LPR, NW, NR, 4><<<grid, block, smem, stream>>>(
                    vx, vy, ids, (const float *) fusion.x_bias, dst,
                    ncols_x, nchannels_y_fd, nrows_x,
                    (uint32_t) nchannels_dst, nchannels_dst_fd, n_pairs, n_y_blocks_fd,
                    stride_row_x, stride_channel_x, stride_col_y, stride_channel_y,
                    stride_col_dst, stride_channel_dst, ids_stride);
            }
            return;
        }
    }
    if constexpr (type == GGML_TYPE_Q8_0) {
        if (!has_ids && ncols_dst >= 1 && ncols_dst <= 4 && nrows_x >= 256 &&
            table_id == MMVQ_PARAMETERS_GCN && warp_size == 64 &&
            fusion.gate == nullptr && fusion.shared_up == nullptr &&
            fusion.gate_bias == nullptr && fusion.gate_scale == nullptr &&
            ((uintptr_t) vx & 3u) == 0u) {
            auto launch_q8 = [&](auto lpr_tag, auto nr_tag, auto ncols_tag, auto ktile_tag) {
                constexpr int LPR    = decltype(lpr_tag)::value;
                constexpr int NW     = 4;
                constexpr int NR     = decltype(nr_tag)::value;
                constexpr int NCOLS  = decltype(ncols_tag)::value;
                constexpr int K_TILE = decltype(ktile_tag)::value;
                constexpr int rpb    = NW * (64 / LPR) * NR;
                const dim3 grid((nrows_x + rpb - 1) / rpb, nchannels_dst, nsamples_dst);
                const dim3 block(64, NW, 1);
                mul_mat_vec_q8_0_gcn<LPR, NW, NR, NCOLS, K_TILE><<<grid, block, 0, stream>>>(
                    vx, vy, (const float *) fusion.x_bias, dst,
                    ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst);
            };
            auto dispatch_shape = [&](auto ncols_tag) {
                constexpr int NCOLS = decltype(ncols_tag)::value;
                if constexpr (NCOLS == 1) {
                    if (blocks_per_row_x <= 16 && nrows_x >= 4096) {
                        launch_q8(std::integral_constant<int, 8>{}, std::integral_constant<int, 4>{}, ncols_tag, std::integral_constant<int, 16>{});
                        return;
                    }
                }
                if (nrows_x >= 6144) {
                    launch_q8(std::integral_constant<int, 16>{}, std::integral_constant<int, 2>{}, ncols_tag, std::integral_constant<int, 64>{});
                } else {
                    launch_q8(std::integral_constant<int, 32>{}, std::integral_constant<int, 1>{}, ncols_tag, std::integral_constant<int, 64>{});
                }
            };
            switch (ncols_dst) {
                case 1: dispatch_shape(std::integral_constant<int, 1>{}); return;
                case 2: dispatch_shape(std::integral_constant<int, 2>{}); return;
                case 3: dispatch_shape(std::integral_constant<int, 3>{}); return;
                case 4: dispatch_shape(std::integral_constant<int, 4>{}); return;
                default: break;
            }
        }
    }
#endif

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            // static, else MSVC lambda capture breaks the constexpr uses below
            static constexpr int c_ncols_dst = 1;

            // Tag types keep the flags compile-time, so __launch_bounds__ matches what is launched.
            const auto launch = [&](auto small_k_tag, auto halve_iters_tag) {
                constexpr bool c_small_k = decltype(small_k_tag)::value;
                // Types the table does not promote would compile a second, identical kernel.
                constexpr bool c_promoted =
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, true) !=
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, false);

                constexpr bool c_halve_iters = decltype(halve_iters_tag)::value && c_promoted;

                const std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst + (fusion.shared_up != nullptr),
                                                                              nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            };

            if (should_use_small_k(c_ncols_dst)) {
                launch(std::true_type{},  std::false_type{});
            } else if (should_halve_iters()) {
                launch(std::false_type{}, std::true_type{});
            } else {
                launch(std::false_type{}, std::false_type{});
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        GGML_ASSERT( !ids || dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc));
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->shared_up) {
            GGML_ASSERT(ids && fusion->gate && fusion->shared_gate && fusion->shared_dst);
            GGML_ASSERT(!fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale);
            GGML_ASSERT(ne11 == 1 && ne03 == 1 && ne13 == 1);
            GGML_ASSERT(fusion->shared_up->type == src0->type && fusion->shared_gate->type == src0->type);
            GGML_ASSERT(ggml_are_same_shape(fusion->shared_up, fusion->shared_gate));
            GGML_ASSERT(ggml_is_contiguous(fusion->shared_up) && ggml_is_contiguous(fusion->shared_gate));
            GGML_ASSERT(fusion->shared_up->ne[0] == ne00 && fusion->shared_up->ne[1] == ne01);
            GGML_ASSERT(fusion->shared_up->nb[1] == nb01 && ggml_is_matrix(fusion->shared_up));
            GGML_ASSERT(fusion->shared_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->shared_dst));
            GGML_ASSERT(fusion->shared_dst->ne[0] == ne0 && fusion->shared_dst->ne[1] == ne2);
            fusion_local.shared_up   = fusion->shared_up->data;
            fusion_local.shared_gate = fusion->shared_gate->data;
            fusion_local.shared_dst  = (float *) fusion->shared_dst->data;
            fusion_local.shared_stride_col_dst = fusion->shared_dst->nb[1] / ts_dst;
        }

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
#if defined(GGML_USE_HIP)
    if (fusion && fusion->situ) {
        const ggml_tensor * gate = fusion->situ->src[0];
        const ggml_tensor * up   = fusion->situ->src[1];
        float beta, lb;
        memcpy(&beta, (const float *) fusion->situ->op_params + 0, sizeof(float));
        memcpy(&lb,   (const float *) fusion->situ->op_params + 1, sizeof(float));
        const dim3 grid((ne10_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE, ne11, ne12 * ne13);
        const uint3 ne12_fd = init_fastdiv_values(ne12);
        quantize_inkling2_situ_q8_1<<<grid, CUDA_QUANTIZE_BLOCK_SIZE, 0, stream>>>(
            (const float *) gate->data, (const float *) up->data, src1_q8_1.get(),
            ne10, gate->nb[1]/sizeof(float), gate->nb[2]/sizeof(float), gate->nb[3]/sizeof(float),
            up->nb[1]/sizeof(float), up->nb[2]/sizeof(float), up->nb[3]/sizeof(float),
            ne10_padded, (uint32_t) ne11, ne12_fd, beta, lb);
    } else
#endif
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, src1_q8_1.get(), ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1.get(), ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
