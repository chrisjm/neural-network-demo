#pragma once

class TriangleMesh;

struct Object2D {
    TriangleMesh* mesh;

    float offsetX;
    float offsetY;
    float scale;
    float rotation; // radians
    float color[3];
};
