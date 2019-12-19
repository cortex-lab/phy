attribute vec3 a_position;
attribute vec4 a_color;

varying vec4 v_color;

void main() {
    vec2 xy = a_position.xy;
    gl_Position = transform(xy);
    gl_Position.z = a_position.z;

    // Set the varyings.
    v_color = a_color;
}
