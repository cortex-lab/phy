#include "markers/%MARKER.glsl"
#include "utils.glsl"

uniform vec4 u_color;
uniform float u_size;

varying float v_mask;

vec4 filled2(float distance, float linewidth, float antialias, vec4 bg_color)
{
    vec4 frag_color;
    float t = linewidth / 2.0 - antialias;
    float signed_distance = distance;
    float border_distance = abs(signed_distance) - t;
    float alpha = border_distance/antialias;
    alpha = exp(-alpha*alpha);

    if (border_distance < 0.0)
        frag_color = bg_color;
    else if (signed_distance < 0.0)
        frag_color = bg_color;
    else {
        if (abs(signed_distance) < (linewidth / 2.0 + antialias)) {
            frag_color = vec4(bg_color.rgb, alpha * bg_color.a);
        }
        else {
            discard;
        }
    }
    return frag_color;
}

void main()
{
    vec2 P = gl_PointCoord.xy - vec2(0.5, 0.5);
    float point_size = u_size + 5.;
    float distance = marker_%MARKER(P * point_size, u_size);
    vec4 color = apply_mask(u_color, v_mask);
    gl_FragColor = filled2(distance, 1.0, 1.0, color);
}
