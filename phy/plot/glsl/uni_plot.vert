#include "utils.glsl"

attribute vec2 a_position;
uniform vec4 u_color;
uniform float u_depth;
attribute float a_signal_index;  // 0..n_signals-1

varying float v_signal_index;

void main() {
    gl_Position = transform(a_position);
    gl_Position.z = u_depth;

    v_signal_index = a_signal_index;
}
