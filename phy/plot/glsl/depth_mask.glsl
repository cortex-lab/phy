float depth_mask(float cluster, float mask, float n_clusters) {
    // Depth and mask.
    float depth = 0.0;
    if (mask > 0.25) {
        depth = -.1 - (cluster + mask) / (n_clusters + 10.);
    }
    return depth;
}
