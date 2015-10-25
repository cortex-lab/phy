#include "antialias/filled.glsl"
#include "markers/%MARKER_TYPE.glsl"

varying vec4 v_color;
varying float v_size;

void main()
{
    vec2 P = gl_PointCoord.xy - vec2(0.5,0.5);
    float point_size = v_size  + 2. * (1.0 + 1.5*1.0);
    float distance = marker_%MARKER_TYPE(P*point_size, v_size);
    gl_FragColor = filled(distance, 1.0, 1.0, v_color);
}
