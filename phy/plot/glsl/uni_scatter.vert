attribute vec2 a_position;

uniform vec4 u_color;
uniform float u_size;
uniform float u_depth;

void main() {
    gl_Position = transform(a_position);
    gl_Position.z = u_depth;

    // Point size as a function of the marker size and antialiasing.
    gl_PointSize = u_size + 5.0;
}
