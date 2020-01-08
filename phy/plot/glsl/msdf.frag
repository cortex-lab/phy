/*
Multi-channel signed distance field
https://github.com/Chlumsky/msdfgen
*/

uniform sampler2D u_tex;
uniform vec4 u_color;
uniform vec2 u_tex_size;
uniform vec2 u_zoom;

varying vec2 v_tex_coords;
varying vec4 v_color;


float median(float r, float g, float b) {
    return max(min(r, g), min(max(r, g), b));
}


float contour(float d, float w) {
    return smoothstep(0.5 - w, 0.5 + w, d);
}


float get_alpha(vec2 uv) {
    vec2 msdfUnit = 4.0 / u_tex_size;
    vec3 sample = texture2D(u_tex, uv).rgb;
    float sigDist = median(sample.r, sample.g, sample.b) - 0.5;
    sigDist *= dot(msdfUnit, 0.5 / fwidth(uv));
    sigDist += 0.5;
    return clamp(sigDist, 0.0, 1.0);
}


float samp(vec2 uv, float w) {
    return contour(get_alpha(uv), w);
}


float supersample(float alpha) {
    // from http://www.java-gaming.org/index.php?PHPSESSID=lvd34ig10qe05pgvq3lj3rh8a4&topic=33612.msg316185#msg316185
    float width = fwidth(alpha);
    // Supersample, 4 extra points
    float dscale = 0.354; // half of 1/sqrt2; you can play with this
    vec2 duv = dscale * (dFdx(v_tex_coords) + dFdy(v_tex_coords));
    vec4 box = vec4(v_tex_coords - duv, v_tex_coords + duv);
    float asum = samp(box.xy, width)
               + samp(box.zw, width)
               + samp(box.xw, width)
               + samp(box.zy, width);
    alpha = (alpha + 0.5 * asum) / 3.0;

    return alpha;
}


void main() {
    // from https://github.com/Chlumsky/msdfgen
    float alpha = get_alpha(v_tex_coords);
    alpha = supersample(alpha);

    // CONTOUR -- does not work well with small font sizes
    // vec3 sample = texture2D(u_tex, v_tex_coords).rgb;
    // float sigDist = median(sample.r, sample.g, sample.b);
    // sigDist = exp(-20 * pow(sigDist - 1, 2));
    // color = mix(vec3(1, 1, 1), u_color.rgb, sigDist);

    gl_FragColor = vec4(v_color.rgb * alpha, v_color.a);
}
