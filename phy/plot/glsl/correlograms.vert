#include "colormaps/color-space.glsl"
#include "color.glsl"
#include "grid.glsl"

attribute vec2 a_position;
attribute float a_box;  // (from 0 to n_rows**2-1)

uniform sampler2D u_cluster_color;

varying vec4 v_color;
varying float v_box;

void main (void)
{
    // ACG/CCG color.
    vec2 rc = row_col(a_box, n_rows);
    if (abs(rc.x - rc.y) < .1) {
        v_color.rgb = get_color(rc.x,
                                u_cluster_color,
                                n_rows);
    }
    else {
        v_color.rgb = vec3(1., 1., 1.);
    }
    v_color.a = 1.;

    vec2 position = pan_zoom_grid(a_position, a_box);
    vec2 box_position = to_box(position, a_box);
    gl_Position = vec4(box_position, 0., 1.);

    // Used for clipping.
    v_position = position;
    v_box = a_box;
}
