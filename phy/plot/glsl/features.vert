#include "colormaps/color-space.glsl"

attribute vec2 a_position;
attribute float a_mask;
attribute vec3 a_box;  // cluster_idx, row, col

uniform float u_size;

varying vec4 v_color;
varying vec3 v_box;
varying float v_size;

uniform sampler2D u_cluster_color;

uniform float n_clusters;

vec3 get_color(float cluster) {
    return texture2D(u_cluster_color,
                     vec2(cluster / (n_clusters - 1.), .5)).xyz;
}

vec3 color_mask(vec3 color, float mask) {
    vec3 hsv = rgb_to_hsv(color);
    // Change the saturation and value as a function of the mask.
    hsv.y = mask;
    hsv.z = .5 * (1. + mask);
    return hsv_to_rgb(hsv);
}

void main (void)
{
    v_size = u_size;

    v_color.rgb = color_mask(get_color(a_box.x), a_mask);
    v_color.a = .5;

    gl_Position = vec4($transform(a_position), 0., 1.);
    gl_PointSize = u_size + 2.0 * (1.0 + 1.5 * 1.0);
}
