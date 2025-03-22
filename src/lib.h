#include <cstdint>

struct Layout {
    uint64_t structure_seed;
    int16_t start_chunk_x;
    int16_t portal_chunk_x;
    int16_t start_chunk_z;
    int16_t portal_chunk_z;
};

extern "C" uint32_t generate_layouts(uint64_t structure_seed_start, uint64_t structure_seed_end, Layout *out, uint32_t out_len);
extern "C" bool test_world_seed(uint64_t world_seed, int32_t start_chunk_x, int32_t start_chunk_z);