#include "color-space.glsl"

vec4 fetch_texture(float index, sampler2D texture, float size) {
    return texture2D(texture, vec2(index / (size - 1.), .5));
}

vec4 apply_mask(vec4 color, float mask) {
    vec3 hsv = rgb_to_hsv(color.rgb);
    hsv.y *= mask;
    hsv.z *= (1. + mask) * .5;
    return vec4(hsv_to_rgb(hsv), color.a);
}
