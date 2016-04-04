#include "markers/%MARKER.glsl"
#include "antialias/filled.glsl"
#include "utils.glsl"

uniform vec4 u_color;
uniform float u_size;

varying float v_mask;

void main()
{
    vec2 P = gl_PointCoord.xy - vec2(0.5, 0.5);
    float point_size = u_size + 5.;
    float distance = marker_%MARKER(P * point_size, u_size);
    vec4 color = apply_mask(u_color, v_mask);
    gl_FragColor = filled(distance, 1.0, 1.0, color);
}
