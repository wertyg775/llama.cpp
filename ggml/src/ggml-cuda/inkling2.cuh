#include "common.cuh"

bool ggml_cuda_inkling2_ctrl_supported(const ggml_tensor * op);

void ggml_cuda_op_inkling2_ctrl(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_inkling2_clip(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// state_dst: write the new state there (a cache view, s_state_q floats per sequence) instead of the tail of dst
void ggml_cuda_op_inkling2_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_dst = nullptr, int64_t s_state_q = 0);

// the number of nodes to skip when the next copy of the new conv state can be fused, else 0
int ggml_cuda_inkling2_conv_state_fusion(const ggml_cgraph * cgraph, int node_idx, float ** state_dst, int64_t * s_state_q);

bool ggml_cuda_inkling2_mix_supported(const ggml_tensor * op);

void ggml_cuda_op_inkling2_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_inkling2_rel_bias_supported(const ggml_tensor * op);

void ggml_cuda_op_inkling2_rel_bias(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_inkling2_situ(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_inkling2_proj_supported(const ggml_tensor * op);

void ggml_cuda_op_inkling2_proj(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
