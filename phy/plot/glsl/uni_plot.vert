#include "utils.glsl"

attribute vec2 a_position;
attribute float a_signal_index;  // 0..n_signals-1
attribute float a_mask;

uniform vec4 u_color;
uniform float u_mask_max;

varying float v_signal_index;
varying float v_mask;

void main() {
    gl_Position = transform(a_position);
    gl_Position.z = get_depth(a_mask, u_mask_max);

    v_signal_index = a_signal_index;
    v_mask = a_mask;
}
