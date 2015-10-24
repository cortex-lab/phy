varying vec4 v_color;
varying float v_signal_index;

void main() {

    // Discard pixels between signals.
    if (fract(v_signal_index) > 0.)
        discard;

    gl_FragColor = v_color;
}
