#include "grid.glsl"

attribute vec2 a_position;

uniform float u_box;

void main() {

    vec2 pos = a_position;
    vec2 position = pan_zoom_grid(pos, u_box);

    gl_Position = vec4(to_box(position, u_box), -1.0, 1.0);

    // Used for clipping.
    v_position = position;
}
