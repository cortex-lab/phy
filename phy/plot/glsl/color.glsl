
vec3 get_color(float cluster, sampler2D texture, float n_clusters) {
    if (cluster < 0)
        return vec3(.5, .5, .5);
    return texture2D(texture, vec2(cluster / (n_clusters - 1.), .5)).xyz;
}

vec3 color_mask(vec3 color, float mask) {
    vec3 hsv = rgb_to_hsv(color);
    // Change the saturation and value as a function of the mask.
    hsv.y *= mask;
    hsv.z *= .5 * (1. + mask);
    return hsv_to_rgb(hsv);
}
