in vec2 a_position;
in vec2 a_tex_coords;

out vec2 v_tex_coords;

void main() {
    gl_Position = transform(a_position);
    v_tex_coords = a_tex_coords;
}
