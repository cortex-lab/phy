#include "color-space.glsl"

vec4 fetch_texture(float index, sampler2D texture, float size) {
    return texture2D(texture, vec2(index / (size - 1.), .5));
}

vec4 apply_mask(vec4 color, float mask) {
    // NOTE: we assume that mask = mask + clu_idx.
    mask = fract(mask);
    vec3 hsv = rgb_to_hsv(color.rgb);
    hsv.y *= mask;
    hsv.z *= (1. + mask) * .5;
    return vec4(hsv_to_rgb(hsv), color.a);
}

float get_depth(float mask, float mask_max) {
    // NOTE: we assume that mask = mask + clu_idx.
    // Background.
    if (mask_max == 0.) {
        return 0.5;
    }

    // Masked elements.
    if (fract(mask) < 0.25) {
        return 0.;
    }

    // Unmasked elements.
    return -0.1 - mask / (mask_max + 10.);
}
