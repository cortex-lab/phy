#include "pan_zoom.glsl"

attribute float a_position;
attribute vec2 a_index;

varying vec2 v_index;

uniform float n_channels;
uniform float n_samples;
uniform float u_scale;

float get_x(float x_index) {
    // 'x_index' is between 0 and nsamples.
    return -1. + 2. * x_index / (float(n_samples) - 1.);
}

float get_y(float y_index, float sample) {
    // 'y_index' is between 0 and n_channels.
    float a = float(u_scale) / float(n_channels);
    float b = -1. + 2. * (y_index + .5) / float(n_channels);
    return a * sample + b;
}

void main() {
    vec2 position = vec2(get_x(a_index.y),
                         get_y(a_index.x, a_position));
    gl_Position = vec4(pan_zoom(position), 0.0, 1.0);
    v_index = a_index;
}
