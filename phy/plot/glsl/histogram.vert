#include "utils.glsl"

attribute vec2 a_position;
attribute float a_hist_index;  // 0..n_hists-1

uniform sampler2D u_hist_colors;
uniform sampler2D u_hist_bounds;
uniform float n_hists;

varying vec4 v_color;
varying float v_hist_index;

void main() {
    vec4 hist_bounds = fetch_texture(a_hist_index,
                                     u_hist_bounds,
                                     n_hists);
    hist_bounds = hist_bounds * 10.;  // NOTE: avoid texture clipping
    gl_Position = transform(a_position);

    v_color = fetch_texture(a_hist_index, u_hist_colors, n_hists);
    v_hist_index = a_hist_index;
}
