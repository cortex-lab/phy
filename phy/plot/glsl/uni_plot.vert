#include "utils.glsl"

in vec2 a_position;
in float a_signal_index;  // 0..n_signals-1
in float a_mask;

uniform vec4 u_color;
uniform float u_mask_max;

out float v_signal_index;
out float v_mask;

void main() {
    gl_Position = transform(a_position);
    gl_Position.z = get_depth(a_mask, u_mask_max);

    v_signal_index = a_signal_index;
    v_mask = a_mask;
}
