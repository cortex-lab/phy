
uniform float n_rows;

varying vec2 v_position;

vec2 to_box(vec2 position, float index) {
    float col = mod(index, n_rows) + 0.5;
    float row = floor(index / n_rows) + 0.5;

    float x = -1.0 + col * (2.0 / n_rows);
    float y = +1.0 - row * (2.0 / n_rows);

    float width = 0.95 / (1.0 * n_rows);
    float height = 0.95 / (1.0 * n_rows);

    return vec2(x + width * position.x,
                y + height * position.y);
}

bool grid_clip(vec2 position) {
    return ((position.x < -0.95) || (position.x > +0.95) ||
            (position.y < -0.95) || (position.y > +0.95));
}
