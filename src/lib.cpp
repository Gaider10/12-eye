#include "lib.h"

#include <climits>
#include <cstdlib>
#include <algorithm>

#include "stronghold_generator/StrongholdGenerator.h"
#include "../cubiomes/finders.h"

extern "C" uint32_t generate_layouts(uint64_t structure_seed_start, uint64_t structure_seed_end, Layout *out, uint32_t out_len) {
    uint32_t count = 0;

    if (count >= out_len) return count;

    stronghold_generator::StrongholdGenerator strongholdGenerator;
    for (uint64_t structure_seed = structure_seed_start; structure_seed < structure_seed_end; structure_seed++) {
        int startChunkX, startChunkZ;
        stronghold_generator::StrongholdGenerator::getFirstPosFast(structure_seed, startChunkX, startChunkZ);
        // stronghold_generator::StrongholdGenerator::getFirstPosOrigin(structure_seed, startChunkX, startChunkZ);
        strongholdGenerator.generate(structure_seed, startChunkX, startChunkZ);
        stronghold_generator::Piece *portalRoomPiece = strongholdGenerator.portalRoomPiece;
        int minChunkX = portalRoomPiece->getWorldX(4,  9) >> 4;
        int maxChunkX = portalRoomPiece->getWorldX(6, 11) >> 4;
        if (minChunkX > maxChunkX) {
            std::swap(minChunkX, maxChunkX);
        }
        int minChunkZ = portalRoomPiece->getWorldZ(4,  9) >> 4;
        int maxChunkZ = portalRoomPiece->getWorldZ(6, 11) >> 4;
        if (minChunkZ > maxChunkZ) {
            std::swap(minChunkZ, maxChunkZ);
        }
        for (int chunkX = minChunkX; chunkX <= maxChunkX; chunkX++) {
            for (int chunkZ = minChunkZ; chunkZ <= maxChunkZ; chunkZ++) {
                stronghold_generator::BoundingBox chunkBox(chunkX << 4, INT_MIN, chunkZ << 4, (chunkX << 4) + 15, INT_MAX, (chunkZ << 4) + 15);

                bool is_first = true;
                for (int i = 0; i < strongholdGenerator.piecesSize - 1; i++) {
                    if (chunkBox.intersects(strongholdGenerator.pieces[i].boundingBox)) {
                        is_first = false;
                        break;
                    }
                }
                if (!is_first) continue;

                out[count++] = Layout{structure_seed, (int16_t)startChunkX, (int16_t)chunkX, (int16_t)startChunkZ, (int16_t)chunkZ};
                if (count >= out_len) return count;
            }
        }
    }

    return count;
}

extern "C" bool test_world_seed(uint64_t world_seed, int32_t start_chunk_x, int32_t start_chunk_z) {
    Generator generator;
    setupGenerator(&generator, MC_NEWEST, 0);
    applySeed(&generator, 0, world_seed);

    StrongholdIter strongholdIter;
    initFirstStronghold(&strongholdIter, generator.mc, world_seed);
    nextStronghold(&strongholdIter, &generator);

    return strongholdIter.pos.x >> 4 == start_chunk_x && strongholdIter.pos.z >> 4 == start_chunk_z;
}