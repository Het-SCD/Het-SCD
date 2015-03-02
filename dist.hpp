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
#ifndef CUDA_HPP
#define CUDA_HPP

#include <vector>
#include "scd.hpp"
#include "graph.hpp"

bool dist_init(std::vector<int> device_ids, int work_group_size, bool profile_flag);
bool dist_transfer_graph(const Graph *g, int *partition, double alpha);
void dist_labels_to_devices(Partition *p);
void dist_improve_labels(double *wcc);
void dist_store_labels();
void dist_labels_from_devices(unsigned int *labels);

#endif
