#include "markers/%MARKER.glsl"
#include "utils.glsl"

varying vec4 v_color;
varying float v_size;
uniform vec2 u_zoom;

void main()
{
    vec2 P = gl_PointCoord.xy - vec2(0.5, 0.5);
    float point_size = v_size + 5.;
    %MARKER_SCALING
    float distance = marker_%MARKER(P * point_size, marker_size);
    gl_FragColor = filled(distance, 1.0, 1.0, v_color);
}
