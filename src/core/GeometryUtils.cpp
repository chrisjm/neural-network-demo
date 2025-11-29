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

bool pointInTriangle(float px, float py,
                     float x0, float y0,
                     float x1, float y1,
                     float x2, float y2) {
    float v0x = x1 - x0;
    float v0y = y1 - y0;
    float v1x = x2 - x0;
    float v1y = y2 - y0;
    float v2x = px - x0;
    float v2y = py - y0;

    float dot00 = v0x * v0x + v0y * v0y;
    float dot01 = v0x * v1x + v0y * v1y;
    float dot02 = v0x * v2x + v0y * v2y;
    float dot11 = v1x * v1x + v1y * v1y;
    float dot12 = v1x * v2x + v1y * v2y;

    float denom = dot00 * dot11 - dot01 * dot01;
    if (denom == 0.0f) return false;

    float invDenom = 1.0f * (1.0f / denom);
    float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    return (u >= 0.0f) && (v >= 0.0f) && (u + v <= 1.0f);
}

bool pointInUnitSquare(float px, float py) {
    return (px >= -0.5f && px <= 0.5f && py >= -0.5f && py <= 0.5f);
}
