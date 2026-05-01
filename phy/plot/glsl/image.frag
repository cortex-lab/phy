uniform sampler2D u_tex;
in vec2 v_tex_coords;

out vec4 FragColor;

void main() {
    FragColor = texture(u_tex, v_tex_coords);
}
