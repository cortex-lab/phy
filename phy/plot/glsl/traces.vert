#include "pan_zoom.glsl"
#include "colormaps/color-space.glsl"
#include "color.glsl"

attribute float a_position;
attribute vec2 a_index;  // (channel_idx, t)
attribute vec2 a_spike;  // (cluster_idx, mask)

uniform sampler2D u_channel_color;
uniform sampler2D u_cluster_color;

uniform float n_channels;
uniform float n_clusters;
uniform float n_samples;
uniform float u_scale;

varying vec3 v_index;  // (channel, cluster, mask)
varying vec3 v_color_channel;
varying vec3 v_color_spike;

float get_x(float x_index) {
    // 'x_index' is between 0 and nsamples.
    return -1. + 2. * x_index / (float(n_samples) - 1.);
}

float get_y(float y_index, float sample) {
    // 'y_index' is between 0 and n_channels.
    float a = float(u_scale) / float(n_channels);
    float b = -1. + 2. * (y_index + .5) / float(n_channels);
    return a * sample + .9 * b;
}

void main() {
    float channel = a_index.x;

    float x = get_x(a_index.y);
    float y = get_y(channel, a_position);
    vec2 position = vec2(x, y);

    gl_Position = vec4(pan_zoom(position), 0.0, 1.0);

    // Spike color as a function of the cluster and mask.
    v_color_spike = color_mask(get_color(a_spike.x,  // cluster_id
                                         u_cluster_color,
                                         n_clusters),
                                   a_spike.y  // mask
                                   );

    // Channel color.
    v_color_channel = get_color(channel,
                                u_channel_color,
                                n_channels);

    // The fragment shader needs to know:
    // * the channel (to discard fragments between channels)
    // * the cluster (for the color)
    // * the mask (for the alpha channel)
    v_index = vec3(channel, a_spike.x, a_spike.y);
}
