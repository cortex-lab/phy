vec4 filled(float distance, float linewidth, float antialias, vec4 bg_color)
{
    vec4 frag_color;
    float t = linewidth/2.0 - antialias;
    float signed_distance = distance;
    float border_distance = abs(signed_distance) - t;
    float alpha = border_distance/antialias;
    alpha = exp(-alpha*alpha);

    if (border_distance < 0.0)
        frag_color = bg_color;
    else if (signed_distance < 0.0)
        frag_color = bg_color;
    else {
        if (abs(signed_distance) < (linewidth/2.0 + antialias)) {
            frag_color = vec4(bg_color.rgb, alpha * bg_color.a);
        }
        else {
            discard;
        }
    }
    return frag_color;
}

vec4 filled(float distance, float linewidth, float antialias, vec4 fg_color, vec4 bg_color)
{
    return filled(distance, linewidth, antialias, fg_color);
}
