
uniform sampler2D u_tex;
uniform vec4 u_color;

varying vec2 v_tex_coords;

void main() {
    gl_FragColor = u_color * texture2D(u_tex, v_tex_coords).x;
}
