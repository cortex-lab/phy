#include "grid.glsl"

attribute vec3 a_position;  // x, y, index

void main() {
    gl_Position = vec4(to_box(a_position.xy, a_position.z),
                       0.0, 1.0);
}
