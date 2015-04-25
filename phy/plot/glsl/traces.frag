
varying vec2 v_index;

void main() {
    gl_FragColor = vec4(1., 1., 1., 1.);
    // Discard vertices between two signals.
    if ((fract(v_index.x) > 0.))
        discard;
}
