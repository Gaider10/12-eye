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

#include "lib.h"

constexpr uint32_t threads_per_block = 256;

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
__managed__ OutputChunkPos outputs[4096];
__managed__ uint64_t outputs_count;

struct PrecompItem {
    // XRSR128 xrsr;
    uint32_t xrsr[4];
    // uint32_t pad;
};
constexpr uint32_t precomp_size = 32 * 20;
__device__ PrecompItem precomp_global[precomp_size];

__device__ void xrsr128_xor(XRSR128 *rng, XRSR128 xor_) {
    rng->lo ^= xor_.lo;
    rng->hi ^= xor_.hi;
}

__device__ void xrsr128_xor(XRSR128 *rng, PrecompItem &xor_) {
    rng->lo ^= ((uint64_t)xor_.xrsr[1] << 32) | xor_.xrsr[0];
    rng->hi ^= ((uint64_t)xor_.xrsr[3] << 32) | xor_.xrsr[2];
}

__device__ int32_t xrsr128_next_bits(XRSR128 *rng, int32_t bits) {
    return (int32_t)((int64_t)xrsr_long(rng) >> (64 - bits));
}

__device__ int64_t xrsr128_nextLong(XRSR128 *rng) {
    return ((int64_t)xrsr128_next_bits(rng, 32) << 32) + (int64_t)xrsr128_next_bits(rng, 32);
}

__device__ void xrsr128_setFeatureSeed(XRSR128 *rng, uint64_t world_seed, int32_t x, int32_t z, int32_t index, int32_t step) {
    xrsr_seed(rng, world_seed);
    int64_t a = xrsr128_nextLong(rng) | 1LL;
    int64_t b = xrsr128_nextLong(rng) | 1LL;
    int64_t decorationSeed = (int64_t)x * a + (int64_t)z * b ^ world_seed;
    int64_t featureSeed = decorationSeed + (int64_t)index + (int64_t)(10000 * step);
    xrsr_seed(rng, featureSeed);
}

__global__ void filter() {
    __shared__ PrecompItem precomp_shared[precomp_size];
    constexpr uint32_t precomp_size_u32 = precomp_size * sizeof(precomp_global[0]) / sizeof(uint32_t);
    for (uint32_t i = 0; i < precomp_size_u32 / threads_per_block; i++) {
        uint32_t index = i * threads_per_block + threadIdx.x;
        reinterpret_cast<uint32_t*>(precomp_shared)[index] = reinterpret_cast<uint32_t*>(precomp_global)[index];
    }
    if (precomp_size_u32 % threads_per_block != 0 && threadIdx.x < precomp_size_u32 % threads_per_block) {
        uint32_t index = precomp_size_u32 / threads_per_block * threads_per_block + threadIdx.x;
        reinterpret_cast<uint32_t*>(precomp_shared)[index] = reinterpret_cast<uint32_t*>(precomp_global)[index];
    }
    __syncthreads();

    uint64_t index = (uint64_t)blockIdx.x * blockDim.x + (uint64_t)threadIdx.x;
    uint64_t world_seed_hi = index & 0xFFFF;
    uint64_t input_chunk_index = index >> 16;
    if (input_chunk_index >= inputs_count) return;

    InputChunkPos inputChunkPos = inputs[input_chunk_index];
    uint64_t world_seed = (world_seed_hi << 48) | inputChunkPos.structure_seed;
    XRSR128 rng;
    xrsr128_setFeatureSeed(&rng, world_seed, (int32_t)inputChunkPos.portal_chunk_x << 4, (int32_t)inputChunkPos.portal_chunk_z << 4, 19, 4);
    uint64_t lo = rng.lo;
    uint64_t hi = rng.hi;

    skip_gpu(&rng);

    xrsr128_xor(&rng, precomp_shared[32 *  0 + ((lo >>       0) & 31)]);
    xrsr128_xor(&rng, precomp_shared[32 *  1 + ((lo >>       5) & 31)]);
    xrsr128_xor(&rng, precomp_shared[32 *  2 + ((lo >>      10) & 31)]);
    xrsr128_xor(&rng, precomp_shared[32 *  3 + ((lo >>      15) & 31)]);
    xrsr128_xor(&rng, precomp_shared[32 *  4 + ((lo >>      20) & 31)]);
    xrsr128_xor(&rng, precomp_shared[32 *  5 + ((lo >>      25) & 31)]);
    xrsr128_xor(&rng, precomp_shared[32 *  6 + ((lo >> 32 +  0) & 31)]);
    xrsr128_xor(&rng, precomp_shared[32 *  7 + ((lo >> 32 +  5) & 31)]);
    xrsr128_xor(&rng, precomp_shared[32 *  8 + ((lo >> 32 + 10) & 31)]);
    xrsr128_xor(&rng, precomp_shared[32 *  9 + ((lo >> 32 + 15) & 31)]);
    xrsr128_xor(&rng, precomp_shared[32 * 10 + ((lo >> 32 + 20) & 31)]);
    xrsr128_xor(&rng, precomp_shared[32 * 11 + ((lo >> 32 + 25) & 31)]);
    xrsr128_xor(&rng, precomp_shared[32 * 12 + ((hi >>       0) & 31)]);
    xrsr128_xor(&rng, precomp_shared[32 * 13 + ((hi >>       5) & 31)]);
    xrsr128_xor(&rng, precomp_shared[32 * 14 + ((hi >>      10) & 31)]);
    xrsr128_xor(&rng, precomp_shared[32 * 15 + ((hi >>      15) & 31)]);
    xrsr128_xor(&rng, precomp_shared[32 * 16 + ((hi >>      20) & 31)]);
    xrsr128_xor(&rng, precomp_shared[32 * 17 + ((hi >>      25) & 31)]);
    xrsr128_xor(&rng, precomp_shared[32 * 18 + ((hi >> 32 +  0) & 31)]);
    xrsr128_xor(&rng, precomp_shared[32 * 19 + ((hi >> 32 +  5) & 31)]);

    for (int j = 0; j < 12; j++) {
        if (xrsr_long(&rng) < 16602070326045573120ULL) {
            return;
        }
    }

    outputs[atomicAdd((unsigned long long *) &outputs_count, 1)] = OutputChunkPos{world_seed, inputChunkPos.start_chunk_x, inputChunkPos.portal_chunk_x, inputChunkPos.start_chunk_z, inputChunkPos.portal_chunk_z};
}

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
   if (code != cudaSuccess) {
      std::fprintf(stderr, "Cuda Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

bool profiling = false;
bool no_layouts = false;

// constexpr uint32_t layout_thread_count = 12;
// constexpr uint32_t thread_inputs_size = inputs_size / layout_thread_count;

// InputChunkPos thread_inputs[layout_thread_count][thread_inputs_size];
// uint64_t thread_inputs_count[layout_thread_count] = {};
// std::thread layout_threads[layout_thread_count];

// void start_layout_threads(uint32_t structure_seed_hi) {
//     uint64_t full_structure_seed_start = (uint64_t)structure_seed_hi << 16;
//     uint64_t full_structure_seed_count = 1 << 16;
//     if (profiling) full_structure_seed_count /= 32;

//     for (uint32_t i = 0; i < layout_thread_count; i++) {
//         layout_threads[i] = std::thread([=](){
//             uint64_t structure_seed_start = full_structure_seed_start + i * full_structure_seed_count / layout_thread_count;
//             uint64_t structure_seed_end = full_structure_seed_start + (i + 1) * full_structure_seed_count / layout_thread_count;
//             thread_inputs_count[i] = generate_layouts(structure_seed_start, structure_seed_end, thread_inputs[i], thread_inputs_size);
//         });
//     }
// }

// void join_layout_threads() {
//     for (int i = 0; i < layout_thread_count; i++) {
//         layout_threads[i].join();
//     }
// }

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

    void start_layout_threads(uint32_t structure_seed_hi) {
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
                uint32_t count = generate_layouts(structure_seed_start, structure_seed_end, thread_inputs.data(), thread_inputs.size());
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

        inputs_count = 0;
        for (auto &thread_data : threads) {
            uint32_t count = thread_data.inputs.size();
            cudaCheckError( cudaMemcpy(inputs + inputs_count, thread_data.inputs.data(), count * sizeof(inputs[0]), cudaMemcpyHostToDevice) );
            inputs_count += count;
        }
    }

private:
    std::vector<LayoutThreadData> threads;
    LayoutThreadPoolState state;
};

void run(uint32_t structure_seed_hi, LayoutThreadPool &layout_thread_pool) {
    outputs_count = 0;
    inputs_count = 0;

    if (layout_thread_pool.get_state() == LayoutThreadPoolState::Empty) {
        layout_thread_pool.start_layout_threads(structure_seed_hi);
        layout_thread_pool.join_layout_threads();
    }

    layout_thread_pool.copy_data();

    // printf("inputs_count = %" PRIu64 " invocations = %" PRIu64 "\n", inputs_count, inputs_count * COUNT16);

    uint32_t thread_count = inputs_count * (1 << 16);
    uint32_t block_count = (thread_count - 1) / threads_per_block + 1;
    filter<<<block_count, threads_per_block>>>();
    cudaCheckError( cudaPeekAtLastError() );

    if (!no_layouts) {
        layout_thread_pool.start_layout_threads(structure_seed_hi + 1);
    }

    cudaCheckError( cudaDeviceSynchronize() );

    auto start = std::chrono::steady_clock::now();

    if (!no_layouts) {
        layout_thread_pool.join_layout_threads();
    }

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1E-9;
    if (elapsed > 0.005) {
        // printf("CPU Thread join took %.3f s\n", elapsed);
    }

    for (uint64_t i = 0; i < outputs_count; i++) {
        OutputChunkPos outputChunkPos = outputs[i];
        // InputChunkPos inputChunkPos = inputs[outputChunkPos.input_index];
        // int64_t world_seed = ((uint64_t)outputChunkPos.world_seed_hi << 48) | ((uint64_t)inputChunkPos.structure_seed_hi << 32) | structure_seed_lo;
        // int64_t world_seed = ((uint64_t)outputChunkPos.world_seed_hi << 48) | inputChunkPos.structure_seed;
        // int startChunkX = 0;
        // int startChunkZ = 0;
        // stronghold_generator::StrongholdGenerator::getFirstPosFast(world_seed, startChunkX, startChunkZ);
        // printf("Seed: %lli Start: %i %i Pos: %i ~ %i\n", world_seed, startChunkX, startChunkZ, (int32_t)inputChunkPos.chunk_x << 4, (int32_t)inputChunkPos.chunk_z << 4);
        bool is_valid = test_world_seed(outputChunkPos.world_seed, outputChunkPos.start_chunk_x, outputChunkPos.start_chunk_z);
        std::printf("Seed: %" PRIi64 " Start: %i %i Pos: %i ~ %i Valid: %s\n", outputChunkPos.world_seed, outputChunkPos.start_chunk_x, outputChunkPos.start_chunk_z, outputChunkPos.portal_chunk_x << 4, outputChunkPos.portal_chunk_z << 4, is_valid ? "YES" : "no");
    }
}

void precompute_bits(PrecompItem *table, unsigned int first, unsigned int count) {
    for (uint64_t bits = 0; bits < ((uint64_t) 1 << count); bits++) {
        uint64_t seed_lo = 0;
        if (first < 64) seed_lo = bits << first;
        uint64_t seed_hi = 0;
        if (first + count > 64) {
            if (first >= 64) seed_hi = bits << (first - 64);
            else seed_hi = bits >> (64 - first);
        }

        XRSR128 rng;
        xrsr128_init(&rng, 0, 0);
        skip_cpu(&rng, seed_lo, seed_hi);
        table[bits] = PrecompItem { { (uint32_t)rng.lo, (uint32_t)(rng.lo >> 32), (uint32_t)rng.hi, (uint32_t)(rng.hi >> 32) } };
    }
}

template<typename T>
void precompute_symbol(T &symbol, uint64_t offset, uint32_t first, uint32_t count) {
    uint32_t table_size = 1 << count;
    PrecompItem *data = new PrecompItem[table_size];
    precompute_bits(data, first, count);
    PrecompItem *address = 0;
    cudaCheckError( cudaGetSymbolAddress((void**) &address, symbol) );
    cudaCheckError( cudaMemcpy(address + offset, data, table_size * sizeof(data[0]), cudaMemcpyHostToDevice) );
    delete[] data;
}

void precompute() {
    auto start = std::chrono::steady_clock::now();

    precompute_symbol(precomp_global, 32 *  0,       0, 5);
    precompute_symbol(precomp_global, 32 *  1,       5, 5);
    precompute_symbol(precomp_global, 32 *  2,      10, 5);
    precompute_symbol(precomp_global, 32 *  3,      15, 5);
    precompute_symbol(precomp_global, 32 *  4,      20, 5);
    precompute_symbol(precomp_global, 32 *  5,      25, 5);
    precompute_symbol(precomp_global, 32 *  6, 32 +  0, 5);
    precompute_symbol(precomp_global, 32 *  7, 32 +  5, 5);
    precompute_symbol(precomp_global, 32 *  8, 32 + 10, 5);
    precompute_symbol(precomp_global, 32 *  9, 32 + 15, 5);
    precompute_symbol(precomp_global, 32 * 10, 32 + 20, 5);
    precompute_symbol(precomp_global, 32 * 11, 32 + 25, 5);
    precompute_symbol(precomp_global, 32 * 12, 64 +  0, 5);
    precompute_symbol(precomp_global, 32 * 13, 64 +  5, 5);
    precompute_symbol(precomp_global, 32 * 14, 64 + 10, 5);
    precompute_symbol(precomp_global, 32 * 15, 64 + 15, 5);
    precompute_symbol(precomp_global, 32 * 16, 64 + 20, 5);
    precompute_symbol(precomp_global, 32 * 17, 64 + 25, 5);
    precompute_symbol(precomp_global, 32 * 18, 96 +  0, 5);
    precompute_symbol(precomp_global, 32 * 19, 96 +  5, 5);

    auto end = std::chrono::steady_clock::now();
    double delta = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1E-9;
    // std::fprintf(stderr, "Precomputing done in %.2fs\n", delta);
}

void bench_layout() {
    uint32_t out_len = 1 << 16;
    std::vector<Layout> out(out_len);

    auto start = std::chrono::steady_clock::now();

    for (uint64_t i = 0;; i++) {
        uint32_t count = generate_layouts(i * out_len, i * out_len + out_len, out.data(), out_len);
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

int main(int argc, char **argv) {
    bool bench = false;

    uint32_t start = UINT32_MAX;
    uint32_t end = UINT32_MAX;
    uint32_t count = UINT32_MAX;
    uint32_t print_interval = 8;
    uint32_t threads = 12;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp("--bench", argv[i]) == 0) {
            bench = true;
        } else if (std::strcmp("--profile", argv[i]) == 0) {
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
        } else {
            std::fprintf(stderr, "Unknwon arg: %s\n", argv[i]);
            return 1;
        }
    }

    if (bench) {
        bench_layout();
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

    std::fprintf(stderr, "start = %"  PRIu32 "\n", start);
    std::fprintf(stderr, "end = %"  PRIu32 "\n", end);
    std::fprintf(stderr, "threads = %"  PRIu32 "\n", threads);
    std::fprintf(stderr, "print_interval = %" PRIu32 "\n", print_interval);

    precompute();

    LayoutThreadPool layout_thread_pool(threads);

    // cudaDeviceProp prop;
    // cudaGetDeviceProperties_v2(&prop, 0);
    // std::printf("prop.sharedMemPerBlock = %zu\n", prop.sharedMemPerBlock);
    // std::printf("prop.sharedMemPerMultiprocessor = %zu\n", prop.sharedMemPerMultiprocessor);
    // std::printf("prop.sharedMemPerBlockOptin = %zu\n", prop.sharedMemPerBlockOptin);

    // cudaCheckError(cudaFuncSetAttribute(filter, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
    // cudaFuncAttributes attr;
    // cudaCheckError(cudaFuncGetAttributes(&attr, filter));
    // std::printf("attr.sharedSizeBytes = %zu\n", attr.sharedSizeBytes);
    // std::printf("attr.constSizeBytes = %zu\n", attr.constSizeBytes);
    // std::printf("attr.localSizeBytes = %zu\n", attr.localSizeBytes);

    auto time_start = std::chrono::steady_clock::now();

    for (uint32_t iter = 0;; iter++) {
        uint32_t structure_seed_hi = start + iter;
        run(structure_seed_hi, layout_thread_pool);
        structure_seed_hi += 1;

        if (print_interval != 0 && (iter + 1) % print_interval == 0) {
            auto time_end = std::chrono::steady_clock::now();
            double delta = std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_start).count() * 1E-9;
            uint64_t seeds_per_run = UINT64_C(1) << 32;
            double sps = print_interval * seeds_per_run / delta;
            std::fprintf(stderr, "%" PRIu32 " %.2f Gsps %.2f h\n", structure_seed_hi, sps * 1E-9, (end - structure_seed_hi) * seeds_per_run / sps / 3600);
            time_start = time_end;
        }

        if (structure_seed_hi == end) break;
    }

    return 0;
}