// -----------------------------------------------------------------------------
// Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
// Distributed under the (new) BSD License.
// -----------------------------------------------------------------------------

float marker_vbar(vec2 P, float marker_size)
{
    return max(abs(P.x) - marker_size / 6.0, abs(P.y) - marker_size / 2.0);
}


float marker_vbar(vec2 P, vec2 marker_size)
{
    return max(abs(P.x) - marker_size.x, abs(P.y) - marker_size.y);
}
