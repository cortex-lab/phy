#include "utils.glsl"

varying vec4 v_color;
varying float v_signal_index;
varying float v_mask;

void main() {

    // Discard pixels between signals.
    if (fract(v_signal_index) > 0.)
        discard;

    gl_FragColor = apply_mask(v_color, v_mask);
}
