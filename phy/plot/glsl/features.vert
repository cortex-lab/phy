
attribute vec2 a_position;
attribute float a_mask;
attribute vec3 a_box;  // cluster_idx, row, col

uniform float u_size;

varying vec4 v_color;
varying vec3 v_box;
varying float v_size;

uniform sampler2D u_cluster_color;

uniform float n_clusters;

vec3 get_color(float cluster) {
    return texture2D(u_cluster_color,
                     vec2(cluster / (n_clusters - 1.), .5)).xyz;
}

void main (void)
{
    v_size = u_size;
    v_color = vec4(get_color(a_box.x), max(1, a_mask));
    // TODO: mask

    gl_Position = vec4($transform(a_position), 0., 1.);
    gl_PointSize = u_size + 2.0 * (1.0 + 1.5 * 1.0);
}
