#include "colormaps/color-space.glsl"
#include "color.glsl"
#include "grid.glsl"
#include "depth_mask.glsl"

attribute vec2 a_position;
attribute float a_mask;
attribute float a_cluster;  // cluster idx
attribute float a_box;  // (from 0 to n_rows**2-1)

uniform float u_size;
uniform float n_clusters;
uniform sampler2D u_cluster_color;

varying vec4 v_color;
varying float v_size;

void main (void)
{
    v_size = u_size;

    v_color.rgb = color_mask(get_color(a_cluster, u_cluster_color, n_clusters),
                             a_mask);
    v_color.a = .5;

    vec2 position = pan_zoom_grid(a_position, a_box);
    vec2 box_position = to_box(position, a_box);

    // Depth as a function of the mask and cluster index.
    float depth = depth_mask(mod(a_box, n_rows), a_mask, n_clusters);

    gl_Position = vec4(box_position, depth, 1.);
    gl_PointSize = u_size + 2.0 * (1.0 + 1.5 * 1.0);

    // Used for clipping.
    v_position = position;
}
