#pragma once

// Convert a world-space point (clip-space x/y in [-1,1]) into the local
// object space used for the original mesh vertices, by inverting the same
// 2D transform that the vertex shader applies (scale + rotation about the
// pivot, then offset).
void worldToLocal(float worldX, float worldY,
                  float offsetX, float offsetY,
                  float scale, float rotation,
                  float& outX, float& outY);

// Simple barycentric test to see if a point lies inside a 2D triangle.
bool pointInTriangle(float px, float py,
                     float x0, float y0,
                     float x1, float y1,
                     float x2, float y2);

// Axis-aligned square centered at the origin in local space, from
// (-0.5, -0.5) to (0.5, 0.5).
bool pointInUnitSquare(float px, float py);
