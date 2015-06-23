#include "colormaps/color-space.glsl"
#include "color.glsl"
#include "pan_zoom.glsl"
#include "depth_mask.glsl"

attribute vec2 a_data;  // position (-1..1), mask
attribute float a_time;  // -1..1
attribute vec2 a_box;  // 0..(n_clusters-1, n_channels-1)

uniform float n_clusters;
uniform float n_channels;
uniform vec2 u_data_scale;
uniform vec2 u_channel_scale;
uniform sampler2D u_channel_pos;
uniform sampler2D u_cluster_color;
uniform float u_overlap;
uniform float u_alpha;

varying vec4 v_color;
varying vec2 v_box;

vec2 get_box_pos(vec2 box) {  // box = (cluster, channel)
    vec2 box_pos = texture2D(u_channel_pos,
                             vec2(box.y / (n_channels - 1.), .5)).xy;
    box_pos = 2. * box_pos - 1.;
    box_pos = box_pos * u_channel_scale;
    // Spacing between cluster boxes.
    float h = 2.5 * u_data_scale.x;
    if (u_overlap < 0.5)
        box_pos.x += h * (box.x - .5 * (n_clusters - 1.)) / n_clusters;
    return box_pos;
}

void main() {
    vec2 pos = u_data_scale * vec2(a_time, a_data.x);  // -1..1
    vec2 box_pos = get_box_pos(a_box);
    v_box = a_box;

    // Depth as a function of the mask and cluster index.
    float depth = depth_mask(a_box.x, a_data.y, n_clusters);

    vec2 x_coeff = vec2(1. / max(n_clusters, 1.), 1.);
    if (u_overlap > 0.5)
        x_coeff.x = 1.;
    // The z coordinate is the depth: it depends on the mask.
    gl_Position = vec4(pan_zoom(x_coeff * pos + box_pos), depth, 1.);

    // Compute the waveform color as a function of the cluster color
    // and the mask.
    v_color.rgb = color_mask(get_color(a_box.x, u_cluster_color, n_clusters),
                             a_data.y);
    v_color.a = u_alpha;
}
