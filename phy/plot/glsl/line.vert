in vec2 a_position;
in vec4 a_color;

out vec4 v_color;

void main() {
    gl_Position = transform(a_position);
    v_color = a_color;
}
