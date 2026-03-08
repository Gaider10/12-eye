#include "xrsr.h"
#include "skip.cuh"
#include <cstdint>
#include <cinttypes>
#include <cstring>
#include <cstdio>
#include <time.h>
#include <vector>
#include <thread>
#include <random>
#include <stdexcept>
#include <array>
#include <cassert>
#include <numeric>
#include <cuda/annotated_ptr>

#include "lib.h"

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
   if (code != cudaSuccess) {
      std::fprintf(stderr, "Cuda Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

constexpr uint32_t skip_threads_per_block = 256;
constexpr uint32_t bloom_threads_per_block = 128;

using InputChunkPos = Layout;
constexpr uint32_t inputs_size = 1 << 16;
__managed__ InputChunkPos inputs[inputs_size];
__managed__ uint32_t inputs_count;

struct OutputChunkPos {
    uint64_t world_seed;
    int16_t start_chunk_x;
    int16_t portal_chunk_x;
    int16_t start_chunk_z;
    int16_t portal_chunk_z;
};
constexpr uint32_t outputs_size = 1024;
__managed__ OutputChunkPos outputs[outputs_size];
__managed__ uint32_t outputs_count;

// 1 GiB
constexpr uint32_t bloom_outputs_size = 67108864;
__managed__ uint32_t bloom_outputs_count;

__device__ void xrsr128_xor(XRSR128 *rng, XRSR128 xor_) {
    rng->lo ^= xor_.lo;
    rng->hi ^= xor_.hi;
}

__device__ int32_t xrsr128_next_bits(XRSR128 *rng, int32_t bits) {
    return (int32_t)((int64_t)xrsr_long(rng) >> (64 - bits));
}

__device__ int64_t xrsr128_nextLong(XRSR128 *rng) {
    return ((int64_t)xrsr128_next_bits(rng, 32) << 32) + (int64_t)xrsr128_next_bits(rng, 32);
}

__device__ uint64_t xrsr128_getDecorationSeed(uint64_t world_seed, int32_t chunk_x, int32_t chunk_z) {
    XRSR128 rng;
    xrsr_seed(&rng, world_seed);
    uint64_t a = xrsr128_nextLong(&rng) | 1;
    uint64_t b = xrsr128_nextLong(&rng) | 1;
    return (chunk_x << 4) * a + (chunk_z << 4) * b ^ world_seed;
}

struct ChunkType {
    uint32_t size;
    uint32_t count;
};

struct ChunkInfo {
    uint32_t bit_offset;
    uint32_t size;
    uint32_t lut_offset;
};

template<ChunkType... ChunksTypes>
struct ChunkedSkip {
    constexpr static std::array<ChunkType, sizeof...(ChunksTypes)> chunk_types {ChunksTypes...};
    constexpr static uint32_t chunked_bits_count = std::accumulate(chunk_types.begin(), chunk_types.end(), 0, [](uint32_t acc, auto &chunk_type){ return acc + chunk_type.size * chunk_type.count; });
    constexpr static uint32_t unchunked_bits_count = 128 - chunked_bits_count;
    constexpr static uint32_t chunks_count = std::accumulate(chunk_types.begin(), chunk_types.end(), 0, [](uint32_t acc, auto &chunk_type){ return acc + chunk_type.count; });
    constexpr static uint32_t operations_count = unchunked_bits_count + chunks_count;

    constexpr ChunkedSkip() : chunks(), operations(), lut_entries() {
        bool chunked_bits[128] = {};

        uint32_t bit_offset = 0;
        uint32_t chunk_index = 0;
        uint32_t lut_offset = 0;
        for (const auto &chunk_type : chunk_types) {
            for (uint32_t i = 0; i < chunk_type.count; i++) {
                if (bit_offset % 64 + chunk_type.size > 64) {
                    bit_offset = bit_offset / 64 * 64 + 64;
                }

                chunks[chunk_index++] = { bit_offset, chunk_type.size, lut_offset };

                for (uint32_t j = 0; j < chunk_type.size; j++) {
                    if (bit_offset >= 128) throw std::invalid_argument("Chunks don't fit");
                    chunked_bits[bit_offset++] = true;
                }

                lut_offset += 1 << chunk_type.size;
            }
        }

        lut_entries = lut_offset;

        uint32_t unchunked_bit_index = 0;
        uint32_t unchunked_bits[unchunked_bits_count + 1];
        for (int i = 0; i < 128; i++) {
            if (chunked_bits[i]) continue;

            unchunked_bits[unchunked_bit_index++] = i;
        }
        if (unchunked_bit_index != unchunked_bits_count) throw std::logic_error("");

        // {
        //     uint32_t bit_index = 0;
        //     uint32_t chunk_index = 0;
        //     for (uint32_t operation_index = 0; operation_index < operations_count; operation_index++) {
        //         if (bit_index * chunks_count <= chunk_index * unchunked_bits_count) {
        //             operations[operation_index] = unchunked_bits[bit_index++];
        //         } else {
        //             operations[operation_index] = 0x80000000 | chunk_index++;
        //         }
        //     }
        // }

        for (uint32_t i = 0; i < unchunked_bits_count; i++) {
            operations[i] = unchunked_bits[i];
        }

        for (uint32_t i = 0; i < chunks_count; i++) {
            operations[unchunked_bits_count + i] = 0x80000000 | i;
        }
    }

    ChunkInfo chunks[chunks_count];
    uint32_t operations[operations_count];
    uint32_t lut_entries;

    void *generate_lut() const {
        auto host_lut = std::make_unique<XRSR128[]>(lut_entries);

        for (const auto &chunk : chunks) {
            for (uint32_t bits = 0; bits < 1 << chunk.size; bits++) {
                uint64_t word = (uint64_t)bits << chunk.bit_offset % 64;
                uint64_t lo = chunk.bit_offset < 64 ? word : 0;
                uint64_t hi = chunk.bit_offset < 64 ? 0 : word;
                XRSR128 rng = {};
                skip_cpu(&rng, lo, hi);
                host_lut[chunk.lut_offset + bits] = rng;
            }
        }

        size_t lut_size = lut_entries * sizeof(XRSR128);
        void *device_lut;
        cudaCheckError(cudaMalloc(&device_lut, lut_size));
        cudaCheckError(cudaMemcpy(device_lut, host_lut.get(), lut_size, cudaMemcpyHostToDevice));

        std::fprintf(stderr, "lut_size = %zu\n", lut_size);
        std::fprintf(stderr, "chunked_bits = %" PRIu32 "\n", chunked_bits_count);
        std::fprintf(stderr, "unchunked_bits = %" PRIu32 "\n", unchunked_bits_count);
        // std::fprintf(stderr, "operations = {\n");
        // for (uint32_t operation : operations) {
        //     if (operation & 0x80000000) {
        //         const auto &chunk = chunks[operation & 0x7FFFFFFF];
        //         std::fprintf(stderr, "  Chunk { .bit_offset = %" PRIu32 ", .size = %" PRIu32 ", .lut_offset = %" PRIu32 " }\n", chunk.bit_offset, chunk.size, chunk.lut_offset);
        //     } else {
        //         uint32_t bit_index = operation;
        //         std::fprintf(stderr, "  Bit { .bit_index = %" PRIu32 " }\n", bit_index);
        //     }
        // }
        // std::fprintf(stderr, "}\n");

        return device_lut;
    }

    __device__ XRSR128 apply(XRSR128 rng, void *lut) const {
        XRSR128 out = {};
        // XRSR128 outs[2] = {};

        #pragma unroll
        for (uint32_t i = 0; i < operations_count; i++) {
            // XRSR128 &out = outs[i % 2];
            uint32_t operation = operations[i];
            if (operation & 0x80000000) {
                const auto &chunk = chunks[operation & 0x7FFFFFFF];
                uint64_t word = chunk.bit_offset < 64 ? rng.lo : rng.hi;
                uint32_t index = (word >> chunk.bit_offset % 64) & ((UINT32_C(1) << chunk.size) - 1);
                ulonglong2 val = __ldg((ulonglong2*)lut + chunk.lut_offset + index);
                out.lo ^= val.x;
                out.hi ^= val.y;
                // uint64_t lo, hi;
                // asm("ld.global.nc.L1::evict_last.v2.u64 { %0, %1 }, [%2];" : "=l"(lo), "=l"(hi) : "l"((XRSR128*)lut + chunk.lut_offset + index));
                // out.lo ^= lo;
                // out.hi ^= hi;
                // XRSR128 *val = (XRSR128*)lut + chunk.lut_offset + index;
                // out.lo ^= val->lo;
                // out.hi ^= val->hi;
                // ulonglong2 *val = (ulonglong2*)lut + chunk.lut_offset + index;
                // out.lo ^= val->x;
                // out.hi ^= val->y;
                // XRSR128 val = *((XRSR128*)lut + chunk.lut_offset + index);
                // out.lo ^= val.lo;
                // out.hi ^= val.hi;
            } else {
                uint32_t bit_index = operation;
                uint64_t word = bit_index < 64 ? rng.lo : rng.hi;
                if ((word >> bit_index % 64) & 1) {
                    out.lo ^= bit_skips[bit_index].lo;
                    out.hi ^= bit_skips[bit_index].hi;
                }
            }
        }

        // out = outs[0];
        // out.lo ^= outs[1].lo;
        // out.hi ^= outs[1].hi;

        return out;
    }
};

#ifdef __CUDA_ARCH__
#define HOST_DEVICE __device__
#else
#define HOST_DEVICE
#endif

// optimized for a 3070
HOST_DEVICE constexpr ChunkedSkip<ChunkType(9, 12), ChunkType(10, 1)> skip_chunked_skip; // 19.02
// HOST_DEVICE constexpr ChunkedSkip<ChunkType(9, 13)> skip_chunked_skip; // 18.82
// HOST_DEVICE constexpr ChunkedSkip<ChunkType(9, 14)> skip_chunked_skip; // 18.05
// HOST_DEVICE constexpr ChunkedSkip<ChunkType(9, 12)> skip_chunked_skip; // 17.21
// HOST_DEVICE constexpr ChunkedSkip<ChunkType(9, 11), ChunkType(10, 2)> skip_chunked_skip; // 13.71
// HOST_DEVICE constexpr ChunkedSkip<ChunkType(9, 10), ChunkType(10, 3)> skip_chunked_skip; // 11.77
HOST_DEVICE constexpr ChunkedSkip<ChunkType(9, 8), ChunkType(10, 3)> decoration_chunked_skip; // 24.92
// HOST_DEVICE constexpr ChunkedSkip<ChunkType(9, 9), ChunkType(10, 2)> decoration_chunked_skip; // 24.85
// HOST_DEVICE constexpr ChunkedSkip<ChunkType(9, 10), ChunkType(10, 1)> decoration_chunked_skip; // 24.61
// HOST_DEVICE constexpr ChunkedSkip<ChunkType(9, 11)> decoration_chunked_skip; // 24.21
// HOST_DEVICE constexpr ChunkedSkip<ChunkType(9, 12)> decoration_chunked_skip; // 17.40 | 23.00
// HOST_DEVICE constexpr ChunkedSkip<ChunkType(10, 1), ChunkType(9, 11)> decoration_chunked_skip; // 17.38 | 22.80
// HOST_DEVICE constexpr ChunkedSkip<ChunkType(9, 10)> decoration_chunked_skip; // 21.51
// HOST_DEVICE constexpr ChunkedSkip<ChunkType(8, 13)> decoration_chunked_skip; // 21.49
// HOST_DEVICE constexpr ChunkedSkip<ChunkType(9, 7), ChunkType(10, 4)> decoration_chunked_skip; // 17.47

__device__ bool test_12_eyes(XRSR128 rng, void *lut) {
    for (int j = 0; j < 12; j++) {
        if (xrsr_long(&rng) < 16602070326045573120ULL) {
            return false;
        }
    }

    return true;
}

__device__ bool test_world_seed_skip(uint64_t world_seed, int32_t chunk_x, int32_t chunk_z, void *lut) {
    uint64_t decoration_seed = xrsr128_getDecorationSeed(world_seed, chunk_x, chunk_z);

    XRSR128 rng;
    xrsr_seed(&rng, decoration_seed + 40019);
    rng = skip_chunked_skip.apply(rng, lut);
    return test_12_eyes(rng, lut);
}

__global__ void filter_skip(void *lut) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t world_seed_hi = index & 0xFFFF;
    uint32_t input_index = index >> 16;
    if (input_index >= inputs_count) return;

    InputChunkPos input = inputs[input_index];
    uint64_t world_seed = ((uint64_t)world_seed_hi << 48) | input.structure_seed;

    if (!test_world_seed_skip(world_seed, input.portal_chunk_x, input.portal_chunk_z, lut)) return;

    uint32_t output_index = atomicAdd(&outputs_count, 1);
    if (output_index < outputs_size) {
        outputs[output_index] = OutputChunkPos{world_seed, input.start_chunk_x, input.portal_chunk_x, input.start_chunk_z, input.portal_chunk_z};
    }
}

template<bool Device>
struct BloomFilter {
    using HostPointer = uint32_t *;
    using DevicePointer = cuda::annotated_ptr<const uint32_t, cuda::access_property::persisting>;
    using Pointer = std::conditional_t<Device, DevicePointer, HostPointer>;
    // using Pointer = HostPointer;

    Pointer data;
    uint32_t mask;

    BloomFilter() = default;

    BloomFilter(uint32_t *data, size_t size) : data(data), mask(size * 8 - 1) {}

    // __host__ __device__ std::array<uint32_t, 1> hash(uint64_t seed) {
    //     return { (uint32_t)(seed >> 32) };
    // }

    __host__ __device__ std::array<uint32_t, 2> hash(uint64_t seed) {
        return { (uint32_t)(seed >> 32), (uint32_t)seed >> 5 };
    }

    __device__ bool get_hash(uint32_t hash) {
        uint32_t bloom_index = hash & mask;
        return (data[bloom_index / 32] >> (bloom_index % 32)) & 1;
    }

    __device__ bool get(uint64_t seed) {
        for (uint32_t hash : hash(seed)) {
            if (!get_hash(hash)) return false;
        }
        return true;
    }

    void set_hash(uint32_t hash) {
        uint32_t bloom_index = hash & mask;
        data[bloom_index / 32] |= UINT32_C(1) << (bloom_index % 32);
    }

    void set(uint64_t seed) {
        for (uint32_t hash : hash(seed)) {
            set_hash(hash);
        }
    }
};

using HostBloomFilter = BloomFilter<false>;
using DeviceBloomFilter = BloomFilter<true>;

using DeviceInputsPtr = cuda::annotated_ptr<InputChunkPos, cuda::access_property::streaming>;
using DeviceOutputsPtr = cuda::annotated_ptr<OutputChunkPos, cuda::access_property::streaming>;
// using DeviceOutputsPtr = cuda::annotated_ptr<uint32_t, cuda::access_property::streaming>;

__global__ void filter_bloom(DeviceBloomFilter bloom_filter, DeviceInputsPtr inputs, DeviceOutputsPtr outputs) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t world_seed_hi = index & 0xFFFF;
    uint32_t input_index = index >> 16;
    if (input_index >= inputs_count) return;

    InputChunkPos input = inputs[input_index];
    uint64_t world_seed = ((uint64_t)world_seed_hi << 48) | input.structure_seed;
    uint64_t decoration_seed = xrsr128_getDecorationSeed(world_seed, input.portal_chunk_x, input.portal_chunk_z);

    if (!bloom_filter.get(decoration_seed)) return;

    uint32_t output_index = atomicAdd(&bloom_outputs_count, 1);
    if (output_index < bloom_outputs_size) {
        outputs[output_index] = OutputChunkPos{world_seed, input.start_chunk_x, input.portal_chunk_x, input.start_chunk_z, input.portal_chunk_z};
    }

    // bool valid = bloom_filter.get(decoration_seed);

    // atomicAdd((unsigned long long *) &outputs_count, valid);

    // uint32_t count = __popc(__ballot_sync(0xFFFFFFFF, valid));
    // __shared__ uint32_t warp_counts[threads_per_block / 32];
    // if (threadIdx.x % 32 == 0) {
    //     warp_counts[threadIdx.x / 32] = count;
    // }
    // __syncthreads();
    // if (threadIdx.x < threads_per_block / 32) {
    //     count = warp_counts[threadIdx.x];
    //     count = __reduce_add_sync(0xFFFFFFFF, count);
    //     if (threadIdx.x == 0 && count) {
    //         atomicAdd((unsigned long long *) &outputs_count, count);
    //     }
    // }

    // __shared__ uint32_t shared_count;
    // // __shared__ OutputChunkPos shared_results[32];
    // __shared__ uint32_t shared_results[32];
    // if (threadIdx.x == 0) shared_count = 0;
    // __syncthreads();
    // uint32_t output_index;
    // if (valid) {
    //     output_index = atomicAdd(&shared_count, 1);
    //     if (output_index < 32) {
    //         // shared_results[output_index] = OutputChunkPos{world_seed, inputChunkPos.start_chunk_x, inputChunkPos.portal_chunk_x, inputChunkPos.start_chunk_z, inputChunkPos.portal_chunk_z};
    //         shared_results[output_index] = index;
    //     } else {
    //         printf("output_index overflow");
    //     }
    // }
    // // uint32_t output_index = atomicAdd(&shared_count, valid);
    // __syncthreads();
    // if (threadIdx.x == 0) {
    //     shared_count = atomicAdd(&outputs_count, shared_count);
    // }
    // __syncwarp();
    // // if (valid) {
    // //     // outputs[shared_count + output_index] = OutputChunkPos{world_seed, inputChunkPos.start_chunk_x, inputChunkPos.portal_chunk_x, inputChunkPos.start_chunk_z, inputChunkPos.portal_chunk_z};
    // //     outputs[shared_count + output_index] = index;
    // // }
    // if (threadIdx.x < std::min(shared_count, 32u)) {
    //     outputs[shared_count + threadIdx.x] = shared_results[threadIdx.x];
    // }
}

__global__ void filter_skip_second(DeviceOutputsPtr inputs, void *lut) {
    uint32_t inputs_len = std::min(bloom_outputs_count, (uint32_t)bloom_outputs_size);
    for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < inputs_len; index += gridDim.x * blockDim.x) {
        OutputChunkPos input = inputs[index];

        if (!test_world_seed_skip(input.world_seed, input.portal_chunk_x, input.portal_chunk_z, lut)) return;

        uint32_t output_index = atomicAdd(&outputs_count, 1);
        if (output_index < outputs_size) {
            outputs[output_index] = input;
        }
    }
}

bool profiling = false;
bool no_layouts = false;
void *device_skip_lut;

struct LayoutThreadData {
    std::thread thread;
    std::vector<InputChunkPos> inputs;

    LayoutThreadData(uint32_t inputs_size) : thread(), inputs(inputs_size) {

    }
};

enum class LayoutThreadPoolState {
    Empty,
    Running,
    HasData,
};

struct LayoutThreadPool {
    LayoutThreadPool(uint32_t thread_count) : threads(), state(LayoutThreadPoolState::Empty) {
        uint32_t thread_inputs_size = inputs_size / thread_count;

        threads.reserve(thread_count);
        for (uint32_t i = 0; i < thread_count; i++) {
            threads.emplace_back(thread_inputs_size);
        }
    }

    LayoutThreadPoolState get_state() const {
        return state;
    }

    void start_layout_threads(uint32_t structure_seed_hi, bool superflat) {
        if (state == LayoutThreadPoolState::Running) throw std::runtime_error("Already Running");

        uint64_t full_structure_seed_start = (uint64_t)structure_seed_hi << 16;
        uint64_t full_structure_seed_count = 1 << 16;
        if (profiling) full_structure_seed_count /= 32;

        for (uint32_t i = 0; i < threads.size(); i++) {
            auto &thread_data = threads[i];
            uint64_t structure_seed_start = full_structure_seed_start + i * full_structure_seed_count / threads.size();
            uint64_t structure_seed_end = full_structure_seed_start + (i + 1) * full_structure_seed_count / threads.size();
            auto &thread_inputs = thread_data.inputs;

            thread_data.thread = std::thread([=, &thread_inputs](){
                thread_inputs.resize(thread_inputs.capacity());
                uint32_t count = generate_layouts(structure_seed_start, structure_seed_end, superflat, thread_inputs.data(), thread_inputs.size());
                thread_inputs.resize(count);
            });
        }

        state = LayoutThreadPoolState::Running;
    }

    void join_layout_threads() {
        if (state != LayoutThreadPoolState::Running) throw std::runtime_error("Not Running");

        for (auto &thread_data : threads) {
            thread_data.thread.join();
        }

        state = LayoutThreadPoolState::HasData;
    }

    void copy_data() {
        if (state != LayoutThreadPoolState::HasData) throw std::runtime_error("Not HasData");

        uint32_t total_count = 0;
        for (auto &thread_data : threads) {
            uint32_t count = thread_data.inputs.size();
            cudaCheckError(cudaMemcpyAsync(inputs + total_count, thread_data.inputs.data(), count * sizeof(inputs[0]), cudaMemcpyHostToDevice));
            total_count += count;
        }
        cudaCheckError(cudaDeviceSynchronize());
        inputs_count = total_count;

        // std::printf("inputs_count = %" PRIu32 "\n", inputs_count);
    }

private:
    std::vector<LayoutThreadData> threads;
    LayoutThreadPoolState state;
};

enum class FilterType {
    Skip,
    Bloom,
};

cudaEvent_t event_start, event_end;

void run(uint32_t structure_seed_hi, bool superflat, LayoutThreadPool &layout_thread_pool, FilterType filter_type, DeviceBloomFilter bloom_filter, void *device_bloom_outputs, bool no_bloom_postfiler, cudaDeviceProp &prop) {
    bool first = layout_thread_pool.get_state() == LayoutThreadPoolState::Empty;

    if (first || !no_layouts) inputs_count = 0;
    outputs_count = 0;
    bloom_outputs_count = 0;

    if (layout_thread_pool.get_state() == LayoutThreadPoolState::Empty) {
        layout_thread_pool.start_layout_threads(structure_seed_hi, superflat);
    }

    if (layout_thread_pool.get_state() == LayoutThreadPoolState::Running) {
        // auto start = std::chrono::steady_clock::now();

        layout_thread_pool.join_layout_threads();

        // auto end = std::chrono::steady_clock::now();
        // double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1E-9;
        // if (elapsed > 0.005) {
        //     printf("CPU Thread join took %.3f s\n", elapsed);
        // }
    }

    if (first || !no_layouts) {
        layout_thread_pool.copy_data();
    }

    if (!no_layouts) {
        layout_thread_pool.start_layout_threads(structure_seed_hi + 1, superflat);
    }

    // printf("inputs_count = %" PRIu64 " invocations = %" PRIu64 "\n", inputs_count, inputs_count * COUNT16);

    cudaCheckError(cudaEventRecord(event_start));

    uint32_t thread_count = inputs_count * (1 << 16);
    uint32_t block_count = thread_count / skip_threads_per_block;
    if (filter_type == FilterType::Skip) {
        filter_skip<<<block_count, skip_threads_per_block>>>(device_skip_lut);
    } else {
        auto bloom_outputs = DeviceOutputsPtr((DeviceOutputsPtr::pointer)device_bloom_outputs);
        {
            uint32_t block_count = thread_count / bloom_threads_per_block;
            filter_bloom<<<block_count, bloom_threads_per_block>>>(bloom_filter, DeviceInputsPtr(inputs), bloom_outputs);
        }
        if (!no_bloom_postfiler) {
            filter_skip_second<<<prop.multiProcessorCount * 16, skip_threads_per_block>>>(bloom_outputs, device_skip_lut);
        }
    }
    cudaCheckError(cudaEventRecord(event_end));
    cudaCheckError(cudaDeviceSynchronize());

    if (filter_type == FilterType::Bloom && bloom_outputs_count >= bloom_outputs_size) {
        std::fprintf(stderr, "bloom_outputs_count >= bloom_outputs_size: %" PRIu32 " >= %" PRIu32 "\n", bloom_outputs_count, bloom_outputs_size);
    }

    if (outputs_count >= outputs_size) {
        std::fprintf(stderr, "outputs_count >= outputs_size: %" PRIu32 " >= %" PRIu32 "\n", outputs_count, outputs_size);
    }

    if (filter_type != FilterType::Bloom || !no_bloom_postfiler) {
        for (uint64_t i = 0; i < outputs_count; i++) {
            OutputChunkPos outputChunkPos = outputs[i];
            bool is_valid = superflat || test_world_seed(outputChunkPos.world_seed, outputChunkPos.start_chunk_x, outputChunkPos.start_chunk_z);
            std::printf("Seed: %" PRIi64 " Start: %i %i Pos: %i ~ %i Valid: %s\n", outputChunkPos.world_seed, outputChunkPos.start_chunk_x, outputChunkPos.start_chunk_z, outputChunkPos.portal_chunk_x << 4, outputChunkPos.portal_chunk_z << 4, is_valid ? "YES" : "no");
        }
    }
}

void bench_layout() {
    uint32_t out_len = 1 << 16;
    std::vector<Layout> out(out_len);

    auto start = std::chrono::steady_clock::now();

    for (uint64_t i = 0;; i++) {
        uint32_t count = generate_layouts(i * out_len, i * out_len + out_len, false, out.data(), out_len);
        std::printf("%" PRIu32 " / %" PRIu32 "\n", count, out_len);

        uint64_t print_interval = 1;
        uint64_t new_i = i + 1;
        if (new_i % print_interval == 0) {
            auto end = std::chrono::steady_clock::now();
            double delta = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1E-9;
            double per_sec = print_interval * out_len / delta;
            std::printf("%" PRIu64 " %.3f s %.3f sps\n", new_i, delta, per_sec);
            start = end;
        }
    }
}

namespace KernelBruteforceDecorationSeeds {
    constexpr uint32_t threads_per_block = 256;
    constexpr uint32_t seeds_per_thread = 16;

    __global__ void kernel(uint32_t hi, void *lut) {
        uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

        uint64_t iter_seed = ((uint64_t)hi << 32) + index * XRSR_GOLDEN_RATIO;
        uint64_t prev_mix = mix64(iter_seed);
        #pragma unroll
        for (int i = 0; i < seeds_per_thread; i++) {
            uint64_t next_mix = mix64(iter_seed + (i + 1) * XRSR_GOLDEN_RATIO);
            XRSR128 rng;
            rng.lo = prev_mix;
            rng.hi = next_mix;
            prev_mix = next_mix;

            rng = decoration_chunked_skip.apply(rng, lut);

            if (!test_12_eyes(rng, lut)) continue;

            uint32_t output_index = atomicAdd(&outputs_count, 1);
        }
    }
}

void bench_decoration(uint32_t print_interval) {
    void *lut = decoration_chunked_skip.generate_lut();

    cudaFuncAttributes attr;
    cudaCheckError(cudaFuncGetAttributes(&attr, KernelBruteforceDecorationSeeds::kernel));
    std::fprintf(stderr, "preferredShmemCarveout = %d\n", attr.preferredShmemCarveout);
    std::fprintf(stderr, "sharedSizeBytes = %zu\n", attr.sharedSizeBytes);

    auto time_start = std::chrono::steady_clock::now();
    uint64_t total_inputs = 0;

    for (uint32_t iter = 0;; iter++) {
        uint64_t seeds_per_run = UINT64_C(1) << 32;
        if (profiling) seeds_per_run /= 32;
        uint64_t threads_per_run = seeds_per_run / KernelBruteforceDecorationSeeds::seeds_per_thread;
        uint32_t blocks_per_run = threads_per_run / KernelBruteforceDecorationSeeds::threads_per_block;
        KernelBruteforceDecorationSeeds::kernel<<<blocks_per_run, KernelBruteforceDecorationSeeds::threads_per_block>>>(iter * XRSR_GOLDEN_RATIO, lut);
        cudaCheckError(cudaDeviceSynchronize());

        total_inputs += seeds_per_run;

        if (print_interval != 0 && (iter + 1) % print_interval == 0) {
            auto time_end = std::chrono::steady_clock::now();
            double delta = std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_start).count() * 1E-9;
            double sps = total_inputs / delta;
            std::fprintf(stderr, "%" PRIu32 " | %.2f Gsps | %.2f d\n",
                iter,
                sps * 1E-9,
                ((UINT64_C(1) << 32) - iter) * seeds_per_run / sps / 3600 / 24
            );
            total_inputs = 0;
            time_start = time_end;
        }
    }

    cudaCheckError(cudaFree(lut));
}

std::vector<uint64_t> generate_random_decoration_seeds() {
    std::fprintf(stderr, "generate_random_decoration_seeds begin\n");

    std::vector<uint64_t> seeds;

    std::mt19937 rng;
    std::uniform_int_distribution<uint64_t> distr(0, UINT64_MAX);

    for (uint32_t i = 0; i < 18446744; i++) {
        seeds.push_back(distr(rng));
    }

    std::fprintf(stderr, "generate_random_decoration_seeds end\n");

    return seeds;
}

bool is_pow_2(auto val) {
    return !(val & (val - 1));
}

struct HostBloomFilters {
    std::unique_ptr<uint32_t[]> data;
    size_t size;

    HostBloomFilters() = default;

    HostBloomFilters(size_t size) : data(std::make_unique<uint32_t[]>(size / 4 * 32)), size(size) {
        if (!is_pow_2(size) || size % 4 != 0) {
            std::fprintf(stderr, "Invalid BloomFilter size: %zu\n", size);
            std::abort();
        }
    }

    HostBloomFilter operator[](size_t index) {
        return HostBloomFilter(data.get() + size / 4 * index, size);
    }
};

HostBloomFilters generate_bloom_filters(const std::vector<uint64_t> &seeds, size_t bloom_filter_size) {
    std::fprintf(stderr, "generate_bloom_filters begin\n");

    HostBloomFilters bloom_filters(bloom_filter_size);

    for (uint64_t seed : seeds) {
        bloom_filters[seed % 32].set(seed);
    }

    std::fprintf(stderr, "generate_bloom_filters end\n");

    return bloom_filters;
}

auto floor_pow2(auto val) {
    for (int i = 1; i < sizeof(val) * 8; i <<= 1) {
        val |= val >> i;
    }
    return val & ~(val >> 1);
}

int main(int argc, char **argv) {
    bool run_bench_layout = false;
    bool run_bench_decoration = false;
    uint32_t start = UINT32_MAX;
    uint32_t end = UINT32_MAX;
    uint32_t count = UINT32_MAX;
    uint32_t print_interval = 8;
    uint32_t threads = std::thread::hardware_concurrency();
    bool superflat = false;
    FilterType filter_type = FilterType::Skip;
    uint32_t bloom_filter_size_override = 0;
    bool no_bloom_postfiler = false;
    bool cycle_bloom_filters = false;
    bool set_persisting_limit = false;
    bool reset_persisting = false;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp("--bench-layout", argv[i]) == 0) {
            run_bench_layout = true;
        } else if (std::strcmp("--bench-decoration", argv[i]) == 0) {
            run_bench_decoration = true;
        }else if (std::strcmp("--profile", argv[i]) == 0) {
            profiling = true;
        } else if (std::strcmp("--no-layouts", argv[i]) == 0) {
            no_layouts = true;
        } else if (std::strcmp("--start", argv[i]) == 0) {
            i += 1;
            if (std::sscanf(argv[i], "%" SCNu32, &start) != 1) {
                std::fprintf(stderr, "Invalid --start: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp("--end", argv[i]) == 0) {
            i += 1;
            if (std::sscanf(argv[i], "%" SCNu32, &end) != 1) {
                std::fprintf(stderr, "Invalid --end: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp("--count", argv[i]) == 0) {
            i += 1;
            if (std::sscanf(argv[i], "%" SCNu32, &count) != 1) {
                std::fprintf(stderr, "Invalid --count: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp("--threads", argv[i]) == 0) {
            i += 1;
            if (std::sscanf(argv[i], "%" SCNu32, &threads) != 1 || threads < 1) {
                std::fprintf(stderr, "Invalid --threads: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp("--print-interval", argv[i]) == 0) {
            i += 1;
            if (std::sscanf(argv[i], "%" SCNu32, &print_interval) != 1) {
                std::fprintf(stderr, "Invalid --print-interval: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp("--superflat", argv[i]) == 0) {
            superflat = true;
        } else if (std::strcmp("--filter", argv[i]) == 0) {
            i += 1;
            if (std::strcmp("skip", argv[i]) == 0) {
                filter_type = FilterType::Skip;
            } else if (std::strcmp("bloom", argv[i]) == 0) {
                filter_type = FilterType::Bloom;
            } else {
                std::fprintf(stderr, "Invalid filter type: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp("--bloom-filter-size", argv[i]) == 0) {
            i += 1;
            if (std::sscanf(argv[i], "%" SCNu32, &bloom_filter_size_override) != 1 || bloom_filter_size_override == 0 || !is_pow_2(bloom_filter_size_override)) {
                std::fprintf(stderr, "Invalid --bloom-filter-size: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp("--no-bloom-postfilter", argv[i]) == 0) {
            no_bloom_postfiler = true;
        } else if (std::strcmp("--cycle-bloom-filters", argv[i]) == 0) {
            cycle_bloom_filters = true;
        } else if (std::strcmp("--set-persisting-limit", argv[i]) == 0) {
            set_persisting_limit = true;
        } else if (std::strcmp("--reset-persisting", argv[i]) == 0) {
            reset_persisting = true;
        } else {
            std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            return 1;
        }
    }

    if (run_bench_layout) {
        bench_layout();
        return 0;
    }

    int device = 0;
    cudaCheckError(cudaSetDevice(device));

    if (run_bench_decoration) {
        bench_decoration(print_interval);
        return 0;
    }

    if (start == UINT32_MAX) {
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
        start = dist(rng);
    }
    if (count != UINT32_MAX) {
        end = start + count;
    }
    if (end == UINT32_MAX) {
        end = start;
    }

    std::fprintf(stderr, "profiling = %s\n", profiling ? "true" : "false");
    std::fprintf(stderr, "no_layouts = %s\n", no_layouts ? "true" : "false");
    std::fprintf(stderr, "start = %"  PRIu32 "\n", start);
    std::fprintf(stderr, "end = %"  PRIu32 "\n", end);
    std::fprintf(stderr, "threads = %"  PRIu32 "\n", threads);
    std::fprintf(stderr, "print_interval = %" PRIu32 "\n", print_interval);
    std::fprintf(stderr, "superflat = %s\n", superflat ? "true" : "false");
    std::fprintf(stderr, "filter_type = %s\n", filter_type == FilterType::Skip ? "skip" : "bloom");
    std::fprintf(stderr, "no_bloom_postfiler = %s\n", no_bloom_postfiler ? "true" : "false");
    std::fprintf(stderr, "set_persisting_limit = %s\n", set_persisting_limit ? "true" : "false");
    std::fprintf(stderr, "clear_persisting = %s\n", reset_persisting ? "true" : "false");

    cudaDeviceProp prop;
    cudaCheckError(cudaGetDeviceProperties(&prop, device));
    std::fprintf(stderr, "l2CacheSize = %d\n", prop.l2CacheSize);
    std::fprintf(stderr, "persistingL2CacheMaxSize = %d\n", prop.persistingL2CacheMaxSize);
    std::fprintf(stderr, "accessPolicyMaxWindowSize = %d\n", prop.accessPolicyMaxWindowSize);
    std::fprintf(stderr, "sharedMemPerMultiprocessor = %zu\n", prop.sharedMemPerMultiprocessor);
    std::fprintf(stderr, "sharedMemPerBlock = %zu\n", prop.sharedMemPerBlock);
    std::fprintf(stderr, "multiProcessorCount = %d\n", prop.multiProcessorCount);

    size_t persistingL2CacheSize;
    cudaCheckError(cudaDeviceGetLimit(&persistingL2CacheSize, cudaLimitPersistingL2CacheSize));
    std::fprintf(stderr, "persistingL2CacheSize = %zu\n", persistingL2CacheSize);

    device_skip_lut = skip_chunked_skip.generate_lut();
    // cudaCheckError(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    // cudaCheckError(cudaFuncSetAttribute(filter_skip, cudaFuncAttributePreferredSharedMemoryCarveout, 0));

    // size_t bloom_filter_size = floor_pow2(std::min(std::min(128 * 1024 * 1024, (int)(prop.l2CacheSize * 0.75)), std::min(prop.persistingL2CacheMaxSize, prop.accessPolicyMaxWindowSize)) / 8 * 8);
    size_t bloom_filter_size = floor_pow2(std::min(128 * 1024 * 1024, (int)(prop.l2CacheSize * 0.75)));
    if (bloom_filter_size_override) bloom_filter_size = bloom_filter_size_override;
    std::fprintf(stderr, "bloom_filter_size = %zu\n", bloom_filter_size);

    if (filter_type == FilterType::Bloom && set_persisting_limit) {
        cudaCheckError(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, std::min((int)bloom_filter_size, prop.persistingL2CacheMaxSize)));
    }

    void *device_bloom_filter_data;
    void *device_bloom_outputs;
    DeviceBloomFilter device_bloom_filter;
    HostBloomFilters host_bloom_filters;

    if (filter_type == FilterType::Bloom) {
        cudaCheckError(cudaMalloc(&device_bloom_filter_data, bloom_filter_size));
        cudaCheckError(cudaMalloc(&device_bloom_outputs, 1024 * 1024 * 1024));

        device_bloom_filter = DeviceBloomFilter(reinterpret_cast<uint32_t*>(device_bloom_filter_data), bloom_filter_size);

        auto seeds = generate_random_decoration_seeds();
        host_bloom_filters = generate_bloom_filters(seeds, bloom_filter_size);
    }

    LayoutThreadPool layout_thread_pool(threads);

    cudaCheckError(cudaEventCreate(&event_start));
    cudaCheckError(cudaEventCreate(&event_end));

    auto time_start = std::chrono::steady_clock::now();

    uint64_t total_inputs_count = 0;
    uint64_t total_bloom_outputs_count = 0;
    double total_kernel_time = 0;

    for (uint32_t iter = 0;; iter++) {
        uint32_t structure_seed_hi = start + iter;
        if (filter_type == FilterType::Bloom && (cycle_bloom_filters || iter == 0)) {
            if (reset_persisting) {
                cudaCheckError(cudaCtxResetPersistingL2Cache());
            }
            cudaCheckError(cudaMemcpyAsync(device_bloom_filter_data, host_bloom_filters[iter % 32].data, bloom_filter_size, cudaMemcpyHostToDevice));
        }
        run(structure_seed_hi, superflat, layout_thread_pool, filter_type, device_bloom_filter, device_bloom_outputs, no_bloom_postfiler, prop);
        structure_seed_hi += 1;
        total_inputs_count += inputs_count;
        total_bloom_outputs_count += bloom_outputs_count;
        float kernel_elapsed;
        cudaCheckError(cudaEventElapsedTime(&kernel_elapsed, event_start, event_end));
        total_kernel_time += kernel_elapsed;

        if (print_interval != 0 && (iter + 1) % print_interval == 0) {
            auto time_end = std::chrono::steady_clock::now();
            double delta = std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_start).count() * 1E-9;
            uint64_t seeds_per_run = UINT64_C(1) << 32;
            double sps = print_interval * seeds_per_run / delta;
            uint64_t total_gpu_seeds = total_inputs_count * 65536;
            std::fprintf(stderr, "%" PRIu32 " | %.2f Gsps | %.2f h | GPU %.2f Gsps | 1 in %.2f | %3.2f %%\n",
                structure_seed_hi,
                sps * 1E-9,
                (end - structure_seed_hi) * seeds_per_run / sps / 3600,
                total_gpu_seeds / delta * 1e-9,
                (double)total_gpu_seeds / total_bloom_outputs_count,
                total_kernel_time * 1e-3 / delta * 1e2
            );
            total_inputs_count = 0;
            total_bloom_outputs_count = 0;
            total_kernel_time = 0;
            time_start = time_end;
        }

        if (structure_seed_hi == end) break;
    }

    if (device_skip_lut) {
        cudaCheckError(cudaFree(device_skip_lut));
    }
    if (device_bloom_filter_data) {
        cudaCheckError(cudaFree(device_bloom_filter_data));
    }
    if (device_bloom_outputs) {
        cudaCheckError(cudaFree(device_bloom_outputs));
    }
    cudaCheckError(cudaEventDestroy(event_start));
    cudaCheckError(cudaEventDestroy(event_end));

    return 0;
}