varying vec4 v_color;
varying vec2 v_box;

void main() {
    if ((fract(v_box.x) > 0.) || (fract(v_box.y) > 0.))
        discard;
    gl_FragColor = v_color;
}
