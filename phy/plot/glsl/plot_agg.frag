// -----------------------------------------------------------------------------
// Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
// Distributed under the (new) BSD License.
// -----------------------------------------------------------------------------

#include "utils.glsl"

// Varyings
// ------------------------------------
in vec4 v_color;
in float v_distance;
in float v_linewidth;
in float v_antialias;
in float v_mask;

out vec4 fragColor;

// Main
// ------------------------------------
void main()
{
    if (v_color.a == 0)  { discard; }
    vec4 strokeColor = stroke(v_distance, v_linewidth, v_antialias, v_color);
    fragColor = apply_mask(gl_FragColor, v_mask);
}
