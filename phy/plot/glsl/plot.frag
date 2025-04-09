#include "utils.glsl"

in vec4 v_color;
in float v_signal_index;
in float v_mask;

out vec4 FragColor

void main() {

    // Discard pixels between signals.
    if (fract(v_signal_index) > 0.)
        discard;

    FragColor = apply_mask(v_color, v_mask);
}
