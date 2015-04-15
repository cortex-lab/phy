uniform vec2 u_zoom;
uniform vec2 u_pan;

vec2 pan_zoom(vec2 position)
{
    return u_zoom * (position + u_pan);
}
