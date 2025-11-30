#version 300 es
precision highp float;
precision highp int;
flat in int vLabel;
flat in int vIndex;
out vec4 FragColor;
uniform vec3 uColorClass0;
uniform vec3 uColorClass1;
uniform int  uSelectedIndex;
void main()
{
    vec2 d = gl_PointCoord - vec2(0.5);
    float r2 = dot(d, d);
    if (r2 > 0.25) discard;
    bool isSelected = (uSelectedIndex >= 0 && vIndex == uSelectedIndex);
    if (isSelected && r2 > 0.16 && r2 < 0.25) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    vec3 color = (vLabel == 0) ? uColorClass0 : uColorClass1;
    FragColor = vec4(color, 1.0);
}
