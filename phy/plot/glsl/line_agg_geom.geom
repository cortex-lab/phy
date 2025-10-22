#version 410
#extension GL_EXT_gpu_shader4 : enable
#extension GL_EXT_geometry_shader4 : enable

uniform mat4 projection;
// uniform float antialias;
// uniform float linewidth;
// uniform float miter_limit;

in in float v_antialias[4][1];
in in float v_linewidth[4][1];
in in float v_miter_limit[4][1];

out out vec2 v_caps;
out out float v_length;
out out vec2 v_texcoord;
out out vec2 v_bevel_distance;

float compute_u(vec2 p0, vec2 p1, vec2 p)
{
    // Projection p' of p such that p' = p0 + u*(p1-p0)
    // Then  u *= length(p1-p0)
    vec2 v = p1 - p0;
    float l = length(v);
    return ((p.x-p0.x)*v.x + (p.y-p0.y)*v.y) / l;
}

float line_distance(vec2 p0, vec2 p1, vec2 p)
{
    // Projection p' of p such that p' = p0 + u*(p1-p0)
    vec2 v = p1 - p0;
    float l2 = v.x*v.x + v.y*v.y;
    float u = ((p.x-p0.x)*v.x + (p.y-p0.y)*v.y) / l2;

    // h is the projection of p on (p0,p1)
    vec2 h = p0 + u*v;

    return length(p-h);
}

void main(void)
{
    float antialias = v_antialias[0][0];
    float linewidth = v_linewidth[0][0];
    float miter_limit = v_miter_limit[0][0];


    // Get the four vertices passed to the shader
    vec2 p0 = gl_PositionIn[0].xy; // start of previous segment
    vec2 p1 = gl_PositionIn[1].xy; // end of previous segment, start of current segment
    vec2 p2 = gl_PositionIn[2].xy; // end of current segment, start of next segment
    vec2 p3 = gl_PositionIn[3].xy; // end of next segment

    // Determine the direction of each of the 3 segments (previous, current, next)
    vec2 v0 = normalize(p1 - p0);
    vec2 v1 = normalize(p2 - p1);
    vec2 v2 = normalize(p3 - p2);

    // Determine the normal of each of the 3 segments (previous, current, next)
    vec2 n0 = vec2(-v0.y, v0.x);
    vec2 n1 = vec2(-v1.y, v1.x);
    vec2 n2 = vec2(-v2.y, v2.x);

    // Determine miter lines by averaging the normals of the 2 segments
    vec2 miter_a = normalize(n0 + n1); // miter at start of current segment
    vec2 miter_b = normalize(n1 + n2); // miter at end of current segment

    // Determine the length of the miter by projecting it onto normal
    vec2 p,v;
    float d;
    float w = linewidth/2.0 + 1.5*antialias;
    v_length = length(p2-p1);

    float length_a = w / dot(miter_a, n1);
    float length_b = w / dot(miter_b, n1);

    float m = miter_limit*linewidth/2.0;

    // Angle between prev and current segment (sign only)
    float d0 = +1.0;
    if( (v0.x*v1.y - v0.y*v1.x) > 0 ) { d0 = -1.0;}

    // Angle between current and next segment (sign only)
    float d1 = +1.0;
    if( (v1.x*v2.y - v1.y*v2.x) > 0 ) { d1 = -1.0; }



    // Generate the triangle strip

    // Cap at start
    if( p0 == p1 ) {
        p = p1 - w*v1 + w*n1;
        gl_Position = projection*vec4(p, 0.0, 1.0);
        v_texcoord = vec2(-w, +w);
        v_caps.x = v_texcoord.x;
    // Regular join
    } else {
        p = p1 + length_a * miter_a;
        gl_Position = projection*vec4(p, 0.0, 1.0);
        v_texcoord = vec2(compute_u(p1,p2,p), +w);
        v_caps.x = 1.0;
    }
    if( p2 == p3 ) v_caps.y = v_texcoord.x;
    else           v_caps.y = 1.0;

    v_bevel_distance.x = +d0*line_distance(p1+d0*n0*w, p1+d0*n1*w, p);
    v_bevel_distance.y =    -line_distance(p2+d1*n1*w, p2+d1*n2*w, p);
    EmitVertex();

    // Cap at start
    if( p0 == p1 ) {
        p = p1 - w*v1 - w*n1;
        v_texcoord = vec2(-w, -w);
        v_caps.x = v_texcoord.x;
    // Regular join
    } else {
        p = p1 - length_a * miter_a;
        v_texcoord = vec2(compute_u(p1,p2,p), -w);
        v_caps.x = 1.0;
    }
    if( p2 == p3 ) v_caps.y = v_texcoord.x;
    else           v_caps.y = 1.0;
    gl_Position = projection*vec4(p, 0.0, 1.0);
    v_bevel_distance.x = -d0*line_distance(p1+d0*n0*w, p1+d0*n1*w, p);
    v_bevel_distance.y =    -line_distance(p2+d1*n1*w, p2+d1*n2*w, p);
    EmitVertex();

    // Cap at end
    if( p2 == p3 ) {
        p = p2 + w*v1 + w*n1;
        v_texcoord = vec2(v_length+w, +w);
        v_caps.y = v_texcoord.x;
    // Regular join
    } else {
        p = p2 + length_b * miter_b;
        v_texcoord = vec2(compute_u(p1,p2,p), +w);
        v_caps.y = 1.0;
    }
    if( p0 == p1 ) v_caps.x = v_texcoord.x;
    else           v_caps.x = 1.0;

    gl_Position = projection*vec4(p, 0.0, 1.0);
    v_bevel_distance.x =    -line_distance(p1+d0*n0*w, p1+d0*n1*w, p);
    v_bevel_distance.y = +d1*line_distance(p2+d1*n1*w, p2+d1*n2*w, p);
    EmitVertex();

    // Cap at end
    if( p2 == p3 ) {
        p = p2 + w*v1 - w*n1;
        v_texcoord = vec2(v_length+w, -w);
        v_caps.y = v_texcoord.x;
    // Regular join
    } else {
        p = p2 - length_b * miter_b;
        v_texcoord = vec2(compute_u(p1,p2,p), -w);
        v_caps.y = 1.0;
    }
    if( p0 == p1 ) v_caps.x = v_texcoord.x;
    else           v_caps.x = 1.0;
    gl_Position = projection*vec4(p, 0.0, 1.0);
    v_bevel_distance.x =    -line_distance(p1+d0*n0*w, p1+d0*n1*w, p);
    v_bevel_distance.y = -d1*line_distance(p2+d1*n1*w, p2+d1*n2*w, p);
    EmitVertex();

    EndPrimitive();
}
