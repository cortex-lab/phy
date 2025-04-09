#include "utils.glsl"

in vec2 a_position;
in float a_hist_index;  // 0..n_hists-1

uniform sampler2D u_color;
uniform float n_hists;

out vec4 v_color;
out float v_hist_index;

void main() {
    gl_Position = transform(a_position);

    v_color = texture(u_color, vec2(a_hist_index / n_hists, 0.0));
    v_hist_index = a_hist_index;
}
