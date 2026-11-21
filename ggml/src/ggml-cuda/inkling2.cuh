#include "common.cuh"

bool ggml_cuda_inkling2_ctrl_supported(const ggml_tensor * op);

void ggml_cuda_op_inkling2_ctrl(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_inkling2_clip(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_inkling2_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_inkling2_mix_supported(const ggml_tensor * op);

void ggml_cuda_op_inkling2_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_inkling2_rel_bias_supported(const ggml_tensor * op);

void ggml_cuda_op_inkling2_rel_bias(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_inkling2_situ(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
