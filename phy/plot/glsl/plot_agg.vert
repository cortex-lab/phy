// -----------------------------------------------------------------------------
// Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
// Distributed under the (new) BSD License.
// -----------------------------------------------------------------------------
#include "utils.glsl"

// Externs
// ------------------------------------
attribute vec3 a_prev;
attribute vec3 a_curr;
attribute vec3 a_next;
attribute float a_id;
attribute vec4 a_color;
attribute float a_mask;
attribute float a_depth;

uniform float u_antialias;
uniform float u_linewidth;
uniform float u_mask_max;

// Varyings
// ------------------------------------
varying float v_antialias;
varying float v_linewidth;
varying float v_distance;
varying vec4  v_color;

varying float v_mask;

vec2 NDC_to_viewport(vec4 position, vec2 viewport)
{
    vec2 p = position.xy/position.w;
    return (p+1.0)/2.0 * viewport;
}

vec4 viewport_to_NDC(vec2 position, vec2 viewport)
{
    return vec4(2.0*(position/viewport) - 1.0, 0.0, 1.0);
}

vec4 viewport_to_NDC(vec2 position, vec2 viewport, float z)
{
    return vec4(2.0*(position/viewport) - 1.0, z, 1.0);
}

// Main
// ------------------------------------
void main (void)
{
    v_linewidth = u_linewidth;
    v_antialias = u_antialias;

    v_color = a_color;

    float id = a_id;

    vec2 p = a_prev.xy;
    vec2 c = a_curr.xy;
    vec2 n = a_next.xy;

    // vec4 Pgl_Position = vec4(0, 0, 0, 1.0);
    // vec4 Cgl_Position = vec4(0, 0, 0, 1.0);
    // vec4 Ngl_Position = vec4(0, 0, 0, 1.0);

    // transform prev/curr/next
    prev = transform(p);
    curr = transform(c);
    next = transform(n);

    vec4 prev_ = vec4(prev, a_prev.z, 1.0);
    vec4 curr_ = vec4(curr, a_curr.z, 1.0);
    vec4 next_ = vec4(next, a_next.z, 1.0);

    // prev/curr/next in viewport coordinates
    vec2 _prev = NDC_to_viewport(prev_, u_window_size);
    vec2 _curr = NDC_to_viewport(curr_, u_window_size);
    vec2 _next = NDC_to_viewport(next_, u_window_size);

    // Compute vertex final position (in viewport coordinates)
    float w = u_linewidth / 2.0 + 1.5 * u_antialias;
    float z;
    vec2 P;
    if( a_curr == a_prev) {
        vec2 v = normalize(_next.xy - _curr.xy);
        vec2 normal = normalize(vec2(-v.y,v.x));
        P = _curr.xy + normal*w*id;
    } else if (a_curr == a_next) {
        vec2 v = normalize(_curr.xy - _prev.xy);
        vec2 normal  = normalize(vec2(-v.y,v.x));
        P = _curr.xy + normal*w*id;
    } else {
        vec2 v0 = normalize(_curr.xy - _prev.xy);
        vec2 v1 = normalize(_next.xy - _curr.xy);
        vec2 normal  = normalize(vec2(-v0.y,v0.x));
        vec2 tangent = normalize(v0+v1);
        vec2 miter   = vec2(-tangent.y, tangent.x);
        float l = abs(w / dot(miter,normal));
        P = _curr.xy + miter*l*sign(id);
    }

    if( abs(id) > 1.5 ) v_color.a = 0.0;

    v_distance = w*id;
    gl_Position = viewport_to_NDC(P, u_window_size, curr_.z / curr_.w);

    gl_Position.z = min(a_depth, get_depth(a_mask, u_mask_max));

    v_mask = a_mask;
}
