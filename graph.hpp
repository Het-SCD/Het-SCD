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
#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <algorithm>

class Graph {
    public:
        const unsigned int n;
        const unsigned int m;
        const double clustering_coef;
        const int *const triangle_deg;
        const int *const triangles;
        const unsigned int *const indices;
        const unsigned int *const adj;

        Graph();
        Graph(unsigned int num_nodes, unsigned int num_edges, unsigned int *adj, unsigned int *indices);
        Graph(unsigned int num_nodes, unsigned int num_edges, unsigned int *adj, unsigned int *indices, int *tri_deg, int *tri);
        Graph(const Graph &that);
        ~Graph();

        static const Graph *read(const char *filename);

        inline const unsigned int *begin_neighbors(unsigned int i) const {
            return adj + indices[i];
        }

        inline const unsigned int *end_neighbors(unsigned int i) const {
            return adj + indices[i + 1];
        }

        inline unsigned int degree(unsigned int i) const {
            return indices[i + 1] - indices[i];
        }

        inline unsigned int max_degree() const {
            unsigned int d = 0;

            for (unsigned int i = 0; i < n; i++) {
                d = std::max(d, degree(i));
            }

            return d;
        }
};


#endif
