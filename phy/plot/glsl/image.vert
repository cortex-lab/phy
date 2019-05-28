attribute vec2 a_position;
attribute vec2 a_tex_coords;

varying vec2 v_tex_coords;

void main() {
    gl_Position = transform(a_position);
    v_tex_coords = a_tex_coords;
}
