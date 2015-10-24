#include "utils.glsl"

attribute vec3 a_position;
attribute float a_signal_index;  // 0..n_signals-1

uniform sampler2D u_signal_bounds;
uniform sampler2D u_signal_colors;
uniform float n_signals;

varying vec4 v_color;
varying float v_signal_index;

void main() {
    // Will be used by the transform.
    vec4 signal_bounds = fetch_texture(a_signal_index,
                                       u_signal_bounds,
                                       n_signals);
    signal_bounds = (2 * signal_bounds - 1);  // See hack in Python.

    vec2 xy = a_position.xy;
    gl_Position = transform(xy);
    gl_Position.z = a_position.z;

    v_color = fetch_texture(a_signal_index, u_signal_colors, n_signals);
    v_signal_index = a_signal_index;
}
