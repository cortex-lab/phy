
attribute vec2 a_position;  // text position
attribute float a_glyph_index;  // glyph index in the text
attribute float a_quad_index;  // quad index in the glyph
attribute float a_char_index;  // index of the glyph in the texture
attribute float a_lengths;
attribute vec2 a_anchor;

uniform vec2 u_glyph_size;  // (w, h)

varying vec2 v_tex_coords;

const float rows = 6;
const float cols = 16;

void main() {
    float w = u_glyph_size.x / u_window_size.x;
    float h = u_glyph_size.y / u_window_size.y;

    float dx = mod(a_quad_index, 2.);
    float dy = 0.;
    if ((2. <= a_quad_index) && (a_quad_index <= 4.)) {
        dy = 1.;
    }

    // Position of the glyph.
    gl_Position = transform(a_position);
    gl_Position.xy = gl_Position.xy + vec2(a_glyph_index * w + dx * w, dy * h);
    // Anchor: the part in [-1, 1] is relative to the text size.
    gl_Position.xy += (a_anchor - 1.) * .5 * vec2(a_lengths * w, h);
    // NOTE: The part beyond [-1, 1] is absolute, so that texts stay aligned.
    gl_Position.xy += (a_anchor - clamp(a_anchor, -1., 1.));

    // Index in the texture
    float i = floor(a_char_index / cols);
    float j = mod(a_char_index, cols);

    // uv position in the texture for the glyph.
    vec2 uv = vec2(j, rows - 1. - i);
    uv /= vec2(cols, rows);

    // Little margin to avoid edge effects between glyphs.
    dx = .01 + .98 * dx;
    dy = .01 + .98 * dy;
    // Texture coordinates for the fragment shader.
    vec2 duv = vec2(dx / cols, dy /rows);

    v_tex_coords = uv + duv;
}
