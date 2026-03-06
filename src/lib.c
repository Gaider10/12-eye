#include "../cubiomes/finders.h"

int test_world_seed(uint64_t world_seed, int32_t start_chunk_x, int32_t start_chunk_z) {
    Generator generator;
    setupGenerator(&generator, MC_NEWEST, 0);
    applySeed(&generator, 0, world_seed);

    StrongholdIter strongholdIter;
    initFirstStronghold(&strongholdIter, generator.mc, world_seed);
    nextStronghold(&strongholdIter, &generator);

    return strongholdIter.pos.x >> 4 == start_chunk_x && strongholdIter.pos.z >> 4 == start_chunk_z;
}