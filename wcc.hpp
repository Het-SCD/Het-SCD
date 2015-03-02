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
#ifndef WCC_HPP
#define WCC_HPP

#include "graph.hpp"
#include "scd.hpp"

double compute_wcc(const Graph *g, double alpha, unsigned int *labels, int *sizes,
        const bool *mask=NULL);

inline static double compute_wcc(const Graph *g, Partition *p, double alpha,
        const bool *mask=NULL) {
    return compute_wcc(g, alpha, p->labels, p->sizes, mask);
}

#endif
