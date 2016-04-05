#include "utils.glsl"

attribute vec2 a_position;
attribute float a_mask;

uniform vec4 u_color;
uniform float u_size;
uniform float u_mask_max;

varying float v_mask;

void main() {
    gl_Position = transform(a_position);
    gl_Position.z = get_depth(a_mask, u_mask_max);

    // Point size as a function of the marker size and antialiasing.
    gl_PointSize = u_size + 5.0;

    v_mask = a_mask;
}
