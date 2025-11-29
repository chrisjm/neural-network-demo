#version 330 core
layout (location = 0) in vec3 aPos;
uniform vec2  uOffset;
uniform float uScale;
uniform float uRotation;
void main()
{
   float c = cos(uRotation);
   float s = sin(uRotation);
   mat3 scaleMat = mat3(
       uScale, 0.0,   0.0,
       0.0,   uScale, 0.0,
       0.0,   0.0,    1.0
   );
   mat3 rotMat = mat3(
       c,  -s,  0.0,
       s,   c,  0.0,
       0.0, 0.0, 1.0
   );
   mat3 transform = rotMat * scaleMat;
   vec2 pivot = vec2(0.0, -0.1666667);
   vec3 localPos = vec3(aPos.xy - pivot, 0.0);
   vec3 rotatedScaled = transform * localPos;
   vec2 worldPos2D = rotatedScaled.xy + pivot + uOffset;
   gl_Position = vec4(worldPos2D, aPos.z, 1.0);
}
