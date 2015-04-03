#include "grid.glsl"

varying vec4 v_color;

void main()
{
    // Clipping.
    if (grid_clip(v_position)) discard;

    gl_FragColor = v_color;
}
