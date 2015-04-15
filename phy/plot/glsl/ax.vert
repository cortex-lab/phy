// xy is the vertex position in NDC
// index is the box index
// ax is 0 for x, 1 for y
attribute vec4 a_position;  // xy, index, ax

#include "grid.glsl"

vec2 pan_zoom(vec2 position, float ax)
{
    if (ax < 0.5)
        return vec2(u_zoom.x * (position.x + n_rows * u_pan.x), position.y);
    else
        return vec2(position.x, u_zoom.y * (position.y + n_rows * u_pan.y));
}

void main() {

    vec2 pos = a_position.xy;
    vec2 position = pan_zoom(pos, a_position.w);

    gl_Position = vec4(to_box(position, a_position.z),
                       0.0, 1.0);

    // Used for clipping.
    v_position = position;
}
