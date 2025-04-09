#include "utils.glsl"

in vec3 a_position;
in vec4 a_color;
in float a_signal_index;  // 0..n_signals-1
in float a_mask;

uniform float u_mask_max;

out vec4 v_color;
out float v_signal_index;
out float v_mask;

void main() {
    vec2 xy = a_position.xy;
    gl_Position = transform(xy);
    gl_Position.z = min(a_position.z, get_depth(a_mask, u_mask_max));

    v_color = a_color;
    v_signal_index = a_signal_index;
    v_mask = a_mask;
}
