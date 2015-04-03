#include "markers/disc.glsl"
#include "antialias/filled.glsl"

varying float v_size;
varying vec4 v_color;
varying vec2 v_position;

void main()
{
    // Clipping.
    if (v_position.x < -0.95) {
        discard;
    }
    else if (v_position.x > +0.95) {
        discard;
    }
    else if (v_position.y < -0.95) {
        discard;
    }
    else if (v_position.y > +0.95) {
        discard;
    }

    vec2 P = gl_PointCoord.xy - vec2(0.5,0.5);
    float point_size = v_size  + 2. * (1.0 + 1.5*1.0);
    float distance = marker_disc(P*point_size, v_size);
    gl_FragColor = filled(distance, 1.0, 1.0, v_color);
}
