#include "grid.glsl"

void main() {
    // Clipping.
    if (grid_clip(v_position, .975)) discard;

    gl_FragColor = vec4(.5, .5, .5, 1.);
}
