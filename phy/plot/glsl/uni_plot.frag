#include "utils.glsl"

uniform vec4 u_color;
in float v_signal_index;
in float v_mask;

out vec4 FragColor;

void main() {

    // Discard pixels between signals.
    if (fract(v_signal_index) > 0.)
        discard;

    FragColor = apply_mask(u_color, v_mask);
}
