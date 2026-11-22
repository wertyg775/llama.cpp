#include "common.cuh"

bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op);

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);

bool ggml_cuda_op_fwht_set_rows_supported(const ggml_tensor * mm, const ggml_tensor * reshape, const ggml_tensor * view, const ggml_tensor * set_rows);
void ggml_cuda_op_fwht_set_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * set_rows);
