uniform float antialias;
uniform float linewidth;
uniform float miter_limit;

in vec2 position;

out float v_antialias[1];
out float v_linewidth[1];
out float v_miter_limit[1];

void main()
{
    v_antialias[0] = antialias;
    v_linewidth[0] = linewidth;
    v_miter_limit[0] = miter_limit;

    gl_Position = transform(position);
}
