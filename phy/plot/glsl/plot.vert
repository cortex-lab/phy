#include "utils.glsl"

attribute vec3 a_position;
attribute vec4 a_color;
attribute float a_signal_index;  // 0..n_signals-1
attribute float a_mask;

uniform float u_mask_max;

varying vec4 v_color;
varying float v_signal_index;
varying float v_mask;

void main() {
    vec2 xy = a_position.xy;
    gl_Position = transform(xy);
    gl_Position.z = min(a_position.z, get_depth(a_mask, u_mask_max));

    v_color = a_color;
    v_signal_index = a_signal_index;
    v_mask = a_mask;
}
