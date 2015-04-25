
varying float v_index_x;
varying vec3 v_color;

void main() {
    gl_FragColor = vec4(v_color, 1.);
    // Discard vertices between two signals.
    if ((fract(v_index_x) > 0.))
        discard;
}
