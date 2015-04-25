
varying vec2 v_index;
varying vec3 v_color_channel;
varying vec3 v_color_spike;

void main() {
    vec3 color;

    // Discard vertices between two channels.
    if ((fract(v_index.x) > 0.))
        discard;

    // Avoid color interpolation at spike boundaries.
    if ((v_index.y >= 0) && (fract(v_index.y) == 0.))
        color = v_color_spike;
    else
        color = v_color_channel;

    gl_FragColor = vec4(color, 1.);
}
