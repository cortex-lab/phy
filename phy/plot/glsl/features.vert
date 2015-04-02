#include "colormaps/color-space.glsl"
#include "color.glsl"

attribute vec2 a_position;
attribute float a_mask;
attribute vec3 a_box;  // cluster_idx, row, col

uniform float u_size;
uniform float n_clusters;
uniform sampler2D u_cluster_color;

varying vec4 v_color;
varying vec3 v_box;
varying float v_size;

void main (void)
{
    v_size = u_size;

    v_color.rgb = color_mask(get_color(a_box.x, u_cluster_color, n_clusters),
                             a_mask);
    v_color.a = .5;

    gl_Position = vec4($transform(a_position), 0., 1.);
    gl_PointSize = u_size + 2.0 * (1.0 + 1.5 * 1.0);
}
