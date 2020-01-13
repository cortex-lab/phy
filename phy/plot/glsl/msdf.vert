
attribute vec2 a_position;  // text position
attribute vec4 a_color;
attribute float a_glyph_index;  // glyph index in the text
attribute float a_quad_index;  // quad index in the glyph
attribute float a_char_index;  // index of the glyph in the texture
attribute float a_lengths;
attribute float a_string_index;  // index of the string

// (1, 1) for lower left, (-1, 1) for lower right,
// (1, -1) for upper left, (-1, -1) for upper right
attribute vec2 a_anchor;

uniform vec2 u_glyph_size;  // (w, h)

varying vec4 v_color;
varying vec2 v_tex_coords;

const float rows = 6;
const float cols = 16;

void main() {
    // Size of one glyph in NDC.
    float w = u_glyph_size.x / u_window_size.x;
    float h = u_glyph_size.y / u_window_size.y;

    // Rectangle vertex displacement (one glyph = one rectangle = 6 vertices)
    float dx = mod(a_quad_index, 2.);
    float dy = 0.;
    if ((2. <= a_quad_index) && (a_quad_index <= 4.)) {
        dy = 1.;
    }

    // Position of the glyph.
    gl_Position = transform(a_position);

    // Displacement based on anchor and glyph index.
    float x = (a_glyph_index + dx) * w;  // relative x position of the vertex
    float y = dy * h;  // relative y position of the vertex
    float xmax = a_lengths * w;  // relative x position of the vertex of the last char
    vec2 origin = .5 * vec2(xmax, h) * (a_anchor - 1);
    gl_Position.xy = gl_Position.xy + origin + vec2(x, y);

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
    v_color = a_color;
}
