#include "common.cuh"

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_argmax_add(ggml_backend_cuda_context & ctx, const ggml_tensor * add_node, ggml_tensor * dst);
