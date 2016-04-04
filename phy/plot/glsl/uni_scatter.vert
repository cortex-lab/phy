attribute vec2 a_position;
attribute float a_mask;

uniform vec4 u_color;
uniform float u_size;
uniform float u_depth;

varying float v_mask;

void main() {
    gl_Position = transform(a_position);
    gl_Position.z = u_depth;

    // Point size as a function of the marker size and antialiasing.
    gl_PointSize = u_size + 5.0;

    v_mask = a_mask;
}
