#include "utils.glsl"

attribute vec3 a_position;
attribute vec4 a_color;
attribute float a_signal_index;  // 0..n_signals-1

varying vec4 v_color;
varying float v_signal_index;

void main() {
    vec2 xy = a_position.xy;
    gl_Position = transform(xy);
    gl_Position.z = a_position.z;

    v_color = a_color;
    v_signal_index = a_signal_index;
}
