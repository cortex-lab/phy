
// NOTE: NOT USED ANYMORE (see msdf.frag instead)

uniform sampler2D u_tex;
uniform vec4 u_color;
uniform vec2 u_zoom;

varying vec2 v_tex_coords;

void main() {
    // Texture scalar.
    float c = texture2D(u_tex, v_tex_coords).x;
    gl_FragColor = vec4(u_color.rgb * c, u_color.a);
}
