#pragma once

#include "Direction.h"

namespace stronghold_generator {
    struct BoundingBox {
        int minX;
        int minY;
        int minZ;
        int maxX;
        int maxY;
        int maxZ;

        BoundingBox(int minX, int minY, int minZ, int maxX, int maxY, int maxZ);

        bool intersects(BoundingBox &boundingBox);
        bool contains(BoundingBox &boundingBox);

        static BoundingBox orientBox(int x, int y, int z, int offsetWidth, int offsetHeight, int offsetDepth, int width, int height, int depth, Direction direction);
    };
}