// TODO: add depth
attribute vec2 a_data;  // position (-1..1), mask
attribute float a_time;  // -1..1
attribute vec2 a_box;  // 0..(n_clusters-1, n_channels-1)

uniform float n_clusters;
uniform float n_channels;
uniform vec2 u_data_scale;
uniform sampler2D u_channel_pos;
uniform sampler2D u_cluster_color;

varying vec4 v_color;
varying vec2 v_box;

// TODO: use VisPy transforms
vec2 get_box_pos(vec2 box) {  // box = (cluster, channel)
    vec2 box_pos = texture2D(u_channel_pos,
                             vec2(box.y / (n_channels - 1.), .5)).xy;
    box_pos = 2. * box_pos - 1.;
    // Spacing between cluster boxes.
    float h = 2.5 * u_data_scale.x;
    // TODO: add superposition
    box_pos.x += h * (box.x - .5 * (n_clusters - 1.));
    return box_pos;
}

vec3 get_color(float cluster) {
    return texture2D(u_cluster_color,
                     vec2(cluster / (n_clusters - 1.), .5)).xyz;
}

void main() {
    vec2 pos = u_data_scale * vec2(a_time, a_data.x);  // -1..1
    vec2 box_pos = get_box_pos(a_box);
    v_box = a_box;
    gl_Position = vec4($transform(pos + box_pos), 0., 1.);

    // Compute the waveform color as a function of the cluster color
    // and the mask.
    float mask = a_data.y;
    // TODO: store the colors in HSV in the texture?
    vec3 rgb = get_color(a_box.x);
    vec3 hsv = $rgb_to_hsv(rgb);
    // Change the saturation and value as a function of the mask.
    hsv.y = mask;
    hsv.z = .5 * (1. + mask);
    v_color.rgb = $hsv_to_rgb(hsv);
    v_color.a = .5;
}
