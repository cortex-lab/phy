// -----------------------------------------------------------------------------
// Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
// Distributed under the (new) BSD License.
// -----------------------------------------------------------------------------

#include "utils.glsl"

// Varyings
// ------------------------------------
varying vec4 v_color;
varying float v_distance;
varying float v_linewidth;
varying float v_antialias;
varying float v_mask;

// Main
// ------------------------------------
void main()
{
    if (v_color.a == 0)  { discard; }
    gl_FragColor = stroke(v_distance, v_linewidth, v_antialias, v_color);
    gl_FragColor = apply_mask(gl_FragColor, v_mask);
}
