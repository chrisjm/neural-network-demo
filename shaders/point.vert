#version 300 es
precision highp float;
precision highp int;
layout (location = 0) in vec2 aPos;
layout (location = 1) in float aLabel;
flat out int vLabel;
flat out int vIndex;
uniform float uPointSize;
void main()
{
    gl_Position = vec4(aPos, 0.0, 1.0);
    vLabel = int(aLabel + 0.5);
    vIndex = gl_VertexID;
    gl_PointSize = uPointSize;
}
