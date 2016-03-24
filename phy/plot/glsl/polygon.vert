attribute vec2 a_position;
uniform vec4 u_color;

void main() {
    gl_Position = transform(a_position);
    gl_Position.z = -.5;
}
