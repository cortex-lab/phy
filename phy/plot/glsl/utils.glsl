vec4 fetch_texture(float index, sampler2D texture, float size) {
    return texture2D(texture, vec2(index / (size - 1.), .5));
}
