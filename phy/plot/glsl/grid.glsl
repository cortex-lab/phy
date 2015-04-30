
uniform float n_rows;
uniform sampler2D u_pan_zoom;  // index (i, j) contains pan_xy, zoom_xy.

varying vec2 v_position;

vec2 row_col(float index, float n_rows) {
    float row = floor(index / n_rows);
    float col = mod(index, n_rows);
    return vec2(row, col);
}

vec4 fetch_pan_zoom(float index) {
    vec2 uv = row_col(index, n_rows);
    // switch because (x, y) = (j, i)
    vec4 pz = texture2D(u_pan_zoom, (uv.yx + .5) / n_rows);
    vec2 pan = pz.xy * 10. - 1.;
    vec2 zoom = pz.zw * 10.;
    return vec4(pan, zoom);
}

vec2 to_box(vec2 position, float index) {
    vec2 rc = row_col(index, n_rows) + 0.5;

    float x = -1.0 + rc.y * (2.0 / n_rows);
    float y = +1.0 - rc.x * (2.0 / n_rows);

    float width = 0.95 / (1.0 * n_rows);
    float height = 0.95 / (1.0 * n_rows);

    return vec2(x + width * position.x,
                y + height * position.y);
}

bool grid_clip(vec2 position, float lim) {
    return ((position.x < -lim) || (position.x > +lim) ||
            (position.y < -lim) || (position.y > +lim));
}

bool grid_clip(vec2 position) {
    return grid_clip(position, .95);
}

vec2 pan_zoom_grid(vec2 position, float index)
{
    vec4 pz = fetch_pan_zoom(index);  // (pan_x, pan_y, zoom_x, zoom_y)
    return pz.zw * (position + n_rows * pz.xy);
}
