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
#ifndef SCD_HPP
#define SCD_HPP

#include "graph.hpp"

#define INVALID_LABEL (~((unsigned int) 0))

typedef struct Partition {
    unsigned int *labels;
    unsigned int num_communities;
    int *sizes;
    int *ext_edges;
    int *int_edges;
} Partition;

Partition *create_initial_partition(const Graph *g);
void compress_labels(const Graph *g, unsigned int *labels, unsigned int *num_communities);
void collect_stats(const Graph *g, Partition *p, const bool *mask=NULL);
void improve_partition(const Graph *g, Partition *p, double alpha, const bool *mask=NULL);

inline static void compress_labels(const Graph *g, Partition *p) {
    compress_labels(g, p->labels, &(p->num_communities));
}


#endif
