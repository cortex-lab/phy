
uniform sampler2D u_tex;
uniform vec4 u_color;

varying vec2 v_tex_coords;

void main() {
    // Texture scalar.
    float c = texture2D(u_tex, v_tex_coords).x;
    gl_FragColor = vec4(u_color.rgb * c, u_color.a);
}
