#include "grid.glsl"
uniform vec4 u_color;

void main() {
    // Clipping.
    if (grid_clip(v_position, .975)) discard;

    gl_FragColor = u_color;
}
