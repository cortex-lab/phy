#include "color-space.glsl"

uniform vec4 u_color;
varying float v_signal_index;
varying float v_mask;

void main() {

    // Discard pixels between signals.
    if (fract(v_signal_index) > 0.)
        discard;

    vec3 hsv = rgb_to_hsv(u_color.rgb);
    hsv.y *= v_mask;
    hsv.z *= (1. + v_mask) * .5;
    gl_FragColor = vec4(hsv_to_rgb(hsv), u_color.a);
}
