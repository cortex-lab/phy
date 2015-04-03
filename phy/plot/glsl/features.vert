#include "colormaps/color-space.glsl"
#include "color.glsl"

attribute vec2 a_position;
attribute float a_mask;
attribute float a_cluster;  // cluster idx
attribute float a_box;  // (from 0 to n_rows**2-1)

uniform float u_size;
uniform float n_rows;
uniform float n_clusters;
uniform sampler2D u_cluster_color;

varying vec4 v_color;
varying vec2 v_position;
varying float v_size;

vec2 to_box(vec2 position, float index) {
    float col = mod(index, n_rows) + 0.5;
    float row = floor(index / n_rows) + 0.5;

    float x = -1.0 + col * (2.0 / n_rows);
    float y = -1.0 + row * (2.0 / n_rows);

    float width = 0.95 / (1.0 * n_rows);
    float height = 0.95 / (1.0 * n_rows);

    return vec2(x + width * position.x,
                y + height * position.y);
}

void main (void)
{
    v_size = u_size;

    v_color.rgb = color_mask(get_color(a_cluster, u_cluster_color, n_clusters),
                             a_mask);
    v_color.a = .5;

    vec2 position = $transform(a_position);
    vec2 box_position = to_box(position, a_box);
    gl_Position = vec4(box_position, 0., 1.);
    gl_PointSize = u_size + 2.0 * (1.0 + 1.5 * 1.0);

    // Used for clipping.
    v_position = position;
}
