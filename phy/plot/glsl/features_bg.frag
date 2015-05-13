#include "grid.glsl"

void main()
{
    // Clipping.
    if (grid_clip(v_position)) discard;

    gl_FragColor = vec4(.25, .3, .35, .25);
}
