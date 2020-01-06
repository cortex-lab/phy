vec3 hsv_to_rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}


vec3 rgb_to_hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}


vec4 fetch_texture(float index, sampler2D texture, float size) {
    return texture2D(texture, vec2(index / (size - 1.), .5));
}


vec4 filled(float distance, float linewidth, float antialias, vec4 bg_color)
{
    vec4 frag_color;
    float t = linewidth/2.0 - antialias;
    float signed_distance = distance;
    float border_distance = abs(signed_distance) - t;
    float alpha = border_distance/antialias;
    alpha = exp(-alpha*alpha);

    if( border_distance < 0.0 ) {
        frag_color = bg_color;
    } else if( signed_distance < 0.0 ) {
        frag_color = bg_color;
    } else {
        frag_color = vec4(bg_color.rgb, alpha * bg_color.a);
    }

    return frag_color;
}


vec4 filled(float distance, float linewidth, float antialias, vec4 fg_color, vec4 bg_color)
{
    return filled(distance, linewidth, antialias, fg_color);
}


vec4 stroke(float distance, float linewidth, float antialias, vec4 fg_color)
{
    vec4 frag_color;
    float t = linewidth/2.0 - antialias;
    float signed_distance = distance;
    float border_distance = abs(signed_distance) - t;
    float alpha = border_distance/antialias;
    alpha = exp(-alpha*alpha);

    if( border_distance < 0.0 )
        frag_color = fg_color;
    else
        frag_color = vec4(fg_color.rgb, fg_color.a * alpha);

    return frag_color;
}


vec4 stroke(float distance, float linewidth, float antialias, vec4 fg_color, vec4 bg_color)
{
    return stroke(distance, linewidth, antialias, fg_color);
}


vec4 apply_mask(vec4 color, float mask) {
    // NOTE: we assume that mask = mask + clu_idx.
    mask = fract(mask);
    vec3 hsv = rgb_to_hsv(color.rgb);
    hsv.y *= mask;
    hsv.z *= (1. + mask) * .5;
    return vec4(hsv_to_rgb(hsv), color.a);
}


float get_depth(float mask, float mask_max) {
    // NOTE: we assume that mask = mask + clu_idx.
    // Background.
    if (mask_max == 0.) {
        return 0.5;
    }

    // Masked elements.
    if (fract(mask) < 0.25) {
        return 0.;
    }

    // Unmasked elements.
    return -0.1 - mask / (mask_max + 10.);
}
