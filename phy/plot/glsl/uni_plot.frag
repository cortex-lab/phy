uniform vec4 u_color;
varying float v_signal_index;

void main() {

    // Discard pixels between signals.
    if (fract(v_signal_index) > 0.)
        discard;

    gl_FragColor = u_color;
}
