attribute vec3 a_position;
attribute vec4 a_color;
attribute float a_size;

varying vec4 v_color;
varying float v_size;

void main() {
    vec2 xy = a_position.xy;
    gl_Position = transform(xy);
    gl_Position.z = a_position.z;

    // Point size as a function of the marker size and antialiasing.
    gl_PointSize = a_size + 5.0;

    // Set the varyings.
    v_color = a_color;
    v_size = a_size;
}
