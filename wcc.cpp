/*
 * Het-SCD: High Quality Community Detection on Heterogenous Platforms 
 * Copyright (C) 2015, S. Heldens
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

// This code implements computation of the WCC metric and is strongly based on
// code from https://github.com/DAMA-UPC/SCD

#include <iostream>

#include "common.hpp"
#include "graph.hpp"
#include "scd.hpp"
#include "wcc.hpp"

using namespace std;

double compute_wcc(const Graph *g, double alpha, unsigned int *labels, int *sizes,
        const bool *mask) {
    timer_start("compute WCC");

    unsigned int n = g->n;
    const unsigned int *const adj = g->adj;
    const unsigned int *const indices = g->indices;
    const int *const triangles = g->triangles;
    const int *const triangle_deg = g->triangle_deg;
    double wcc = 0.0;
    int count = 0;

#pragma omp parallel
    {
        unsigned int neighbors_size = 0;
        unsigned int neighbors_cap = 1024;
        unsigned int *neighbors = new unsigned int[1024];

#pragma omp for schedule(dynamic, 32) reduction(+:wcc,count)
        for (unsigned int p = 0; p < n; p++) {

            if (mask != NULL && !mask[p]) {
                continue;
            }

            count++;
            unsigned int label = labels[p];

            if (label == INVALID_LABEL) {
                continue;
            }

            int tri = triangles[p];
            int tri_deg = triangle_deg[p];
            int int_tri = 0;
            int int_tri_deg = 0;

            int size = sizes[label];

            if (size <= 2 || tri_deg == 0) {
                continue;
            }

            unsigned int deg = indices[p + 1] - indices[p];

            if (deg > neighbors_cap) {
                neighbors_cap = deg;
                delete[] neighbors;
                neighbors = new unsigned int[neighbors_cap];
            }

            neighbors_size = 0;

            for (unsigned int i = indices[p]; i != indices[p + 1]; i++) {
                const unsigned int q = adj[i];

                if (labels[q] == label) {
                    neighbors[neighbors_size++] = q;
                }
            }

            for (unsigned int i = 0; i < neighbors_size; i++) {
                unsigned int q = neighbors[i];

                if (label != labels[q]) {
                    continue;
                }

                unsigned int a = 0;
                unsigned int b = indices[q];
                bool found = false;

                unsigned int a_end = neighbors_size;
                unsigned int b_end = indices[q + 1];

                while (a != a_end && b != b_end) {
                    unsigned int x = neighbors[a];
                    unsigned int y = adj[b];

                    if (x == y) {
                        found = true;
                        int_tri++;
                    }

                    if (x <= y) {
                        a++;
                    }

                    if (x >= y) {
                        b++;
                    }
                }

                if (found) {
                    int_tri_deg++;
                }
            }

            double nom = int_tri * tri_deg;
            double denom = tri * (tri_deg + alpha * (size - 1 - int_tri_deg));

            if (denom != 0.0) {
                wcc += nom / denom;
            }
        }
    }

    timer_end();

    return count > 0 ? wcc / count : 0;
}
