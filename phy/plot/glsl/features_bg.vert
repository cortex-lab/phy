#include "grid.glsl"

attribute vec2 a_position;
attribute float a_box;  // (from 0 to n_rows**2-1)

void main (void)
{
    vec2 position = pan_zoom_grid(a_position, a_box);
    vec2 box_position = to_box(position, a_box);

    gl_Position = vec4(box_position, 0., 1.);
    gl_PointSize = 3.0;

    // Used for clipping.
    v_position = position;
}
