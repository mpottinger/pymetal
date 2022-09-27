//
//  Voronoi.metal
//  Voronoi
//
//  Created by asd on 21/04/2019.
//  Copyright © 2019 voicesync. All rights reserved.
// check syntax
// xcrun -sdk macosx metal -c voronoi.metal (-o voronoi.air)
// xcrun -sdk macosx metallib voronoi.air -o voronoi.metallib


#include <metal_stdlib>
using namespace metal;


typedef uint32_t color; // aa bb gg rr  32 bit color

inline int sqr(int x) { return x * x; }

inline int distance_sqr(uint4 pntcol, int x, int y) {
    return sqr(x - pntcol.x) + sqr(y - pntcol.y);
}

color genPixel(uint i, uint j, uint count, device uint4* pointscolors) {
    auto pntcol=pointscolors;
    int ind = -1, dist = distance_sqr(*pntcol, i, j);

    for (uint it = 1; it < count; it++, pntcol++) { // find closest point to x,y

        int d = distance_sqr(*pntcol, i, j);

        if (d < 4) { ind=-1; break; } // draw point?

        if (d < dist) {
            dist = d;
            ind = (int)it;
        }
    }

    return 0xff000000 | ((ind > -1) ?  pointscolors[ind].z : 0);
}

kernel void Voronoi(device color*pixels[[buffer(0)]],
                    device uint4*pointscolors[[buffer(1)]], // x,y,color, count

                    uint2 position [[thread_position_in_grid]],
                    uint2 tpg[[threads_per_grid]])
{
    uint x=position.x, y=position.y, width=tpg.x, n_points=pointscolors[0][3];
    pixels[x + y * width] = genPixel(x, y, n_points, pointscolors);
}


