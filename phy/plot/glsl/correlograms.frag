#include "grid.glsl"

varying vec4 v_color;
varying float v_box;

void main()
{
    // Clipping.
    if (grid_clip(v_position)) discard;
    if (fract(v_box) > 0.) discard;

    gl_FragColor = v_color;
}
