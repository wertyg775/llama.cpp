#include "batch-copy.cuh"
#include "common.cuh"
#include "ggml-cpp.h"

#include <climits>
#include <mutex>
#include <vector>

#if defined(GGML_USE_HIP)
struct ggml_hip_host_alloc_entry {
    uintptr_t host;
    uintptr_t dev;
    size_t    size;
};

static std::mutex                              ggml_hip_host_allocs_mutex;
static std::vector<ggml_hip_host_alloc_entry>  ggml_hip_host_allocs;

void ggml_hip_register_host_alloc(void * host_ptr, size_t size) {
    if (host_ptr == nullptr || size == 0) {
        return;
    }
    void * dev_ptr = nullptr;
    if (cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0) != cudaSuccess || dev_ptr == nullptr) {
        (void) cudaGetLastError();
        return;
    }
    std::lock_guard<std::mutex> lock(ggml_hip_host_allocs_mutex);
    ggml_hip_host_allocs.push_back({ (uintptr_t) host_ptr, (uintptr_t) dev_ptr, size });
}

void ggml_hip_unregister_host_alloc(void * host_ptr) {
    if (host_ptr == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(ggml_hip_host_allocs_mutex);
    const uintptr_t addr = (uintptr_t) host_ptr;
    for (size_t i = 0; i < ggml_hip_host_allocs.size(); ++i) {
        if (ggml_hip_host_allocs[i].host == addr) {
            ggml_hip_host_allocs.erase(ggml_hip_host_allocs.begin() + i);
            return;
        }
    }
}

static __global__ void ggml_hip_copy_bytes_kernel(unsigned char * __restrict__ dst, const unsigned char * __restrict__ src, size_t size) {
    const size_t tid    = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    const uintptr_t align = (uintptr_t) dst | (uintptr_t) src;

    size_t tail = 0;
    if ((align & 15) == 0) {
        const size_t n16 = size >> 4;
        uint4 * __restrict__ d16 = (uint4 *) dst;
        const uint4 * __restrict__ s16 = (const uint4 *) src;
        for (size_t i = tid; i < n16; i += stride) {
            d16[i] = s16[i];
        }
        tail = n16 << 4;
    } else if ((align & 3) == 0) {
        const size_t n4 = size >> 2;
        uint32_t * __restrict__ d4 = (uint32_t *) dst;
        const uint32_t * __restrict__ s4 = (const uint32_t *) src;
        for (size_t i = tid; i < n4; i += stride) {
            d4[i] = s4[i];
        }
        tail = n4 << 2;
    }

    for (size_t i = tail + tid; i < size; i += stride) {
        dst[i] = src[i];
    }
}

bool ggml_hip_copy_pinned_async(int device, void * dst, const void * src, size_t size, cudaMemcpyKind kind, cudaStream_t stream) {
    // keep large transfers (such as prefill weight uploads) on SDMA
    constexpr size_t max_kernel_copy_bytes = 2 * 1024 * 1024;
    if (size == 0) {
        return true;
    }
    if (size > max_kernel_copy_bytes) {
        return false;
    }
    if (kind != cudaMemcpyHostToDevice && kind != cudaMemcpyDeviceToHost) {
        return false;
    }

    const uintptr_t host_addr = kind == cudaMemcpyHostToDevice ? (uintptr_t) src : (uintptr_t) dst;
    uintptr_t host_dev_addr = 0;
    {
        std::lock_guard<std::mutex> lock(ggml_hip_host_allocs_mutex);
        for (const auto & e : ggml_hip_host_allocs) {
            if (host_addr >= e.host && host_addr - e.host <= e.size && size <= e.size - (host_addr - e.host)) {
                host_dev_addr = e.dev + (host_addr - e.host);
                break;
            }
        }
    }
    if (host_dev_addr == 0) {
        return false;
    }

    unsigned char * k_dst = kind == cudaMemcpyHostToDevice ? (unsigned char *) dst : (unsigned char *) host_dev_addr;
    const unsigned char * k_src = kind == cudaMemcpyHostToDevice ? (const unsigned char *) host_dev_addr : (const unsigned char *) src;

    const uintptr_t align = (uintptr_t) k_dst | (uintptr_t) k_src;
    const size_t elem_size = (align & 15) == 0 ? 16 : ((align & 3) == 0 ? 4 : 1);
    const size_t n_elems = (size + elem_size - 1) / elem_size;
    constexpr int threads = 256;
    const unsigned int blocks = (unsigned int) std::min<size_t>(256, std::max<size_t>(1, (n_elems + threads - 1) / threads));

    ggml_cuda_set_device(device);
    ggml_hip_copy_bytes_kernel<<<blocks, threads, 0, stream>>>(k_dst, k_src, size);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
#endif

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
struct ggml_cuda_batch_copy_descriptor {
    const unsigned char * src;
    unsigned char * dst;
    size_t size;
};

static __global__ void ggml_cuda_batch_copy_kernel(const ggml_cuda_batch_copy_descriptor * copies, size_t count) {
    // Blocks along y cooperate on each descriptor.
    const size_t index = blockIdx.x;
    if (index >= count) {
        return;
    }
    const size_t first = (size_t) blockIdx.y * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.y * blockDim.x;
    const auto copy = copies[index];
    // Use 16-, 8-, 4-, or 2-byte copies when both pointers meet that alignment; threads copy separate chunks.
    const uintptr_t alignment = (uintptr_t) copy.src | (uintptr_t) copy.dst;
    size_t tail = 0;
    if (alignment % 16 == 0) {
        for (size_t i = first; i < copy.size / 16; i += stride) {
            ggml_cuda_memcpy_1<16>(copy.dst + 16*i, copy.src + 16*i);
        }
        tail = copy.size / 16 * 16;
    } else if (alignment % 8 == 0) {
        for (size_t i = first; i < copy.size / 8; i += stride) {
            ggml_cuda_memcpy_1<8>(copy.dst + 8*i, copy.src + 8*i);
        }
        tail = copy.size / 8 * 8;
    } else if (alignment % 4 == 0) {
        for (size_t i = first; i < copy.size / 4; i += stride) {
            ggml_cuda_memcpy_1<4>(copy.dst + 4*i, copy.src + 4*i);
        }
        tail = copy.size / 4 * 4;
    } else if (alignment % 2 == 0) {
        for (size_t i = first; i < copy.size / 2; i += stride) {
            ggml_cuda_memcpy_1<2>(copy.dst + 2*i, copy.src + 2*i);
        }
        tail = copy.size / 2 * 2;
    }
    // Copy trailing bytes, or the entire range when no wider alignment is available.
    for (size_t i = tail + first; i < copy.size; i += stride) {
        ggml_cuda_memcpy_1<1>(copy.dst + i, copy.src + i);
    }
}

struct ggml_cuda_batch_copy_mapping {
    void * host = nullptr;
    unsigned char * device = nullptr;

    ggml_cuda_batch_copy_mapping(void * host, size_t size) {
        const cudaError_t err = cudaHostRegister(host, size, cudaHostRegisterMapped | cudaHostRegisterPortable);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
            return;
        }
        this->host = host;
        if (cudaHostGetDevicePointer((void **) &device, host, 0) != cudaSuccess) {
            (void) cudaGetLastError();
            device = nullptr;
            return;
        }
    }

    ggml_cuda_batch_copy_mapping(const ggml_cuda_batch_copy_mapping &) = delete;
    ggml_cuda_batch_copy_mapping & operator=(const ggml_cuda_batch_copy_mapping &) = delete;

    ~ggml_cuda_batch_copy_mapping() {
        if (host) {
            // Release only the registration acquired by this object.
            CUDA_CHECK(cudaHostUnregister(host));
        }
    }
};
#endif

static bool ggml_cuda_copy_tensor_n(int device, cudaMemcpyKind direction,
        const ggml_tensor * const * tensors, const void * const * data, const size_t * offsets,
        const size_t * sizes, size_t n_tensors) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(device);
    GGML_UNUSED(direction);
    GGML_UNUSED(tensors);
    GGML_UNUSED(data);
    GGML_UNUSED(offsets);
    GGML_UNUSED(sizes);
    GGML_UNUSED(n_tensors);
    return false;
#else
    const bool to_host = direction == cudaMemcpyDeviceToHost;
    // Provisional dispatch values based on experiments.
    constexpr size_t min_batch_reads = 128;
    constexpr size_t max_kernel_read_bytes = 128 * 1024;
    // Collect destination ranges; repeated sources are allowed.
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
    ranges.reserve(n_tensors);
    size_t payload_bytes = 0;
    uintptr_t begin = UINTPTR_MAX;
    uintptr_t end = 0;
    // Number of non-empty reads whose size is at most max_kernel_read_bytes.
    size_t n_reads_small = 0;
    for (size_t i = 0; i < n_tensors; ++i) {
        if (sizes[i] == 0) {
            continue;
        }
        const uintptr_t address = (uintptr_t) data[i];
        GGML_ASSERT(address <= UINTPTR_MAX - sizes[i]);
        GGML_ASSERT(payload_bytes <= SIZE_MAX - sizes[i]);
        payload_bytes += sizes[i];
        n_reads_small += sizes[i] <= max_kernel_read_bytes;
        begin = std::min(begin, address);
        end = std::max(end, address + sizes[i]);
        const uintptr_t destination = to_host ? address : (uintptr_t) tensors[i]->data + offsets[i];
        GGML_ASSERT(destination <= UINTPTR_MAX - sizes[i]);
        ranges.emplace_back(destination, destination + sizes[i]);
    }
    if (ranges.empty()) {
        return true;
    }
    // Total number of non-empty reads.
    const size_t n_reads = ranges.size();
    if (n_reads > SIZE_MAX / sizeof(ggml_cuda_batch_copy_descriptor) || n_reads > INT_MAX) {
        return false;
    }
    if (n_reads < 2 || (n_reads < min_batch_reads && n_reads_small == n_reads)) {
        return false;
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i) {
        // Preserve the ordered copy path when destinations overlap.
        if (ranges[i].first < ranges[i - 1].second) {
            return false;
        }
    }
    // Limit gaps in the CPU span without underflow when sources overlap.
    if (end - begin > payload_bytes && end - begin - payload_bytes > payload_bytes) {
        return false;
    }
    // Register and map the CPU span for direct kernel access.
    ggml_cuda_batch_copy_mapping mapping((void *) begin, end - begin);
    if (!mapping.host) {
        return false;
    }
    // Reads selected for the kernel; zero means all reads use DMA.
    size_t n_reads_kernel = mapping.device && n_reads_small >= min_batch_reads ? n_reads_small : 0;
    // Keep descriptor storage alive until the final stream synchronization.
    constexpr size_t max_batch_descriptors = 128 * 1024;
    const size_t bytes = std::min(n_reads_kernel, max_batch_descriptors) * sizeof(ggml_cuda_batch_copy_descriptor);
    ggml_backend_buffer_ptr descriptors(n_reads_kernel ? ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device), bytes) : nullptr);
    if (!descriptors) {
        n_reads_kernel = 0;
    }
    const auto host_type = ggml_backend_cuda_host_buffer_type();
    ggml_backend_buffer_ptr host_descriptors(n_reads_kernel ? ggml_backend_buft_alloc_buffer(host_type, bytes) : nullptr);
    if (!host_descriptors || ggml_backend_buffer_get_type(host_descriptors.get()) != host_type) {
        n_reads_kernel = 0;
    }
    if (n_reads_kernel) {
        auto * copies = static_cast<ggml_cuda_batch_copy_descriptor *>(ggml_backend_buffer_get_base(host_descriptors.get()));
        auto * device_copies = static_cast<ggml_cuda_batch_copy_descriptor *>(ggml_backend_buffer_get_base(descriptors.get()));
        const int threads = 256;
        const size_t nsm = ggml_cuda_info().devices[device].nsm;
        size_t i = 0;
        for (size_t first = 0; first < n_reads_kernel; first += max_batch_descriptors) {
            const size_t n_batch = std::min(n_reads_kernel - first, max_batch_descriptors);
            size_t count = 0;
            for (; i < n_tensors && count < n_batch; ++i) {
                if (sizes[i] == 0 || sizes[i] > max_kernel_read_bytes) {
                    continue;
                }
                auto * host = mapping.device + ((uintptr_t) data[i] - begin);
                auto * tensor = (unsigned char *) tensors[i]->data + offsets[i];
                copies[count++] = {to_host ? tensor : host, to_host ? host : tensor, sizes[i]};
            }
            const size_t blocks_per_copy = std::min<size_t>(UINT16_MAX, std::max<size_t>(1, (nsm + n_batch - 1) / n_batch));
            const dim3 blocks((unsigned int) n_batch, (unsigned int) blocks_per_copy);
            // Stream ordering keeps each kernel ahead of the next descriptor overwrite.
            CUDA_CHECK(cudaMemcpyAsync(device_copies, copies, n_batch * sizeof(copies[0]), cudaMemcpyHostToDevice, cudaStreamPerThread));
            ggml_cuda_batch_copy_kernel<<<blocks, threads, 0, cudaStreamPerThread>>>(device_copies, n_batch);
            CUDA_CHECK(cudaGetLastError());
            if (first + n_batch < n_reads_kernel) {
                // Wait before refilling the pinned host descriptor buffer.
                CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
            }
        }
    }
    // Keep the CPU span pinned for DMA too; overlapping destinations already declined above.
    for (size_t i = 0; i < n_tensors; ++i) {
        if (sizes[i] && (!n_reads_kernel || sizes[i] > max_kernel_read_bytes)) {
            void * host = const_cast<void *>(data[i]);
            void * tensor = (char *) tensors[i]->data + offsets[i];
            CUDA_CHECK(cudaMemcpyAsync(to_host ? host : tensor, to_host ? tensor : host, sizes[i], direction, cudaStreamPerThread));
        }
    }
    // Wait for transfers before releasing the CPU mapping and descriptor buffers.
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    return true;
#endif
}

bool ggml_cuda_get_tensor_n(int device,
        const ggml_tensor * const * tensors, void * const * data, const size_t * offsets,
        const size_t * sizes, size_t n_tensors) {
    return ggml_cuda_copy_tensor_n(device, cudaMemcpyDeviceToHost, tensors, data, offsets, sizes, n_tensors);
}

bool ggml_cuda_set_tensor_n(int device,
        ggml_tensor * const * tensors, const void * const * data, const size_t * offsets,
        const size_t * sizes, size_t n_tensors) {
    return ggml_cuda_copy_tensor_n(device, cudaMemcpyHostToDevice, tensors, data, offsets, sizes, n_tensors);
}
