
attribute vec2 a_position;
attribute vec4 a_color;

uniform float u_size;

varying vec4 v_color;
varying float v_size;

void main (void)
{
    v_size = u_size;
    v_color = a_color;

    gl_Position = vec4($transform(a_position), 0., 1.);
    gl_PointSize = u_size + 2.0 * (1.0 + 1.5 * 1.0);
}
