#include "markers/%MARKER.glsl"
#include "antialias/filled.glsl"

varying vec4 v_color;
varying float v_size;

void main()
{
    vec2 P = gl_PointCoord.xy - vec2(0.5, 0.5);
    float point_size = v_size + 5.;
    float distance = marker_%MARKER(P *point_size, v_size);
    gl_FragColor = filled(distance, 1.0, 1.0, v_color);
}
