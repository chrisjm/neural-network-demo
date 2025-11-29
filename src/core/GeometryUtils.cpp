#include "GeometryUtils.h"

#include <cmath>

void worldToLocal(float worldX, float worldY,
                  float offsetX, float offsetY,
                  float scale, float rotation,
                  float& outX, float& outY) {
    const float pivotX = 0.0f;
    const float pivotY = -0.1666667f;

    // Undo the translation by offset.
    float x = worldX - offsetX;
    float y = worldY - offsetY;

    // Move into pivot space.
    x -= pivotX;
    y -= pivotY;

    // Undo rotation: vertex shader uses rotation matrix with +rotation, so
    // we apply -rotation here.
    float c = std::cos(-rotation);
    float s = std::sin(-rotation);
    float rx = c * x - s * y;
    float ry = s * x + c * y;

    // Undo uniform scale (avoid divide by zero).
    if (scale != 0.0f) {
        rx /= scale;
        ry /= scale;
    }

    // Move back out of pivot space.
    rx += pivotX;
    ry += pivotY;

    outX = rx;
    outY = ry;
}

bool pointInTriangle(float pointX, float pointY,
                     float v0x, float v0y,
                     float v1x, float v1y,
                     float v2x, float v2y) {
    // Edges from vertex 0 to vertices 1 and 2.
    float edge0x = v1x - v0x;
    float edge0y = v1y - v0y;
    float edge1x = v2x - v0x;
    float edge1y = v2y - v0y;

    // Vector from vertex 0 to the point.
    float pointVecX = pointX - v0x;
    float pointVecY = pointY - v0y;

    float dot00 = edge0x * edge0x + edge0y * edge0y;
    float dot01 = edge0x * edge1x + edge0y * edge1y;
    float dot02 = edge0x * pointVecX + edge0y * pointVecY;
    float dot11 = edge1x * edge1x + edge1y * edge1y;
    float dot12 = edge1x * pointVecX + edge1y * pointVecY;

    float denom = dot00 * dot11 - dot01 * dot01;
    if (denom == 0.0f) return false;

    float invDenom = 1.0f / denom;
    float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    return (u >= 0.0f) && (v >= 0.0f) && (u + v <= 1.0f);
}

bool pointInUnitSquare(float pointX, float pointY) {
    return (pointX >= -0.5f && pointX <= 0.5f && pointY >= -0.5f && pointY <= 0.5f);
}
