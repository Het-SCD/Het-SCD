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

// This code implements the SCD algorithm and is strongly based on
// code from https://github.com/DAMA-UPC/SCD
#include <algorithm>
#include <iostream>
#include <map>
#include <omp.h>

#include "common.hpp"
#include "graph.hpp"
#include "scd.hpp"

using namespace std;

void compress_labels(const Graph *g, unsigned int *labels, unsigned int *num_communities) {
    timer_start("compress labels");

    unsigned int n = g->n;
    unsigned int num_labels = 0;
    map<unsigned int, unsigned int> label_map;

    for (unsigned int p = 0; p < n; p++) {
        unsigned int l = labels[p];

        if (l == INVALID_LABEL) {
            l = num_labels++;
        } else if (label_map.count(l)) {
            l = label_map[l] = num_labels++;
        } else {
            l = label_map[l];
        }

        labels[p] = l;
    }

    *num_communities = num_labels;
    timer_end();
}

void collect_stats(const Graph *g, Partition *par, const bool *mask) {
    timer_start("collect community statistics");

    const unsigned int n = g->n;
    const unsigned int *const indices = g->indices;
    const unsigned int *const adj = g->adj;

    const unsigned int num_comms = par->num_communities;
    unsigned int *const labels = par->labels;
    int *const ext_edges = par->ext_edges;
    int *const int_edges = par->int_edges;
    int *const sizes = par->sizes;

#pragma omp parallel
    {

#pragma omp for schedule(static)
        for (unsigned int i = 0; i < num_comms; i++) {
            sizes[i] = 0;
            ext_edges[i] = 0;
            int_edges[i] = 0;
        }

#pragma omp barrier

#pragma omp for schedule(dynamic,512)
        for (unsigned int p = 0; p < n; p++) {
            unsigned int label_p = labels[p];

            if ((mask != NULL && !mask[p]) || label_p == INVALID_LABEL) {
                continue;
            }

            int ext_count = 0;
            int int_count = 0;

            for (unsigned int i = indices[p]; i != indices[p + 1]; i++) {
                unsigned int q = adj[i];
                unsigned int label_q = labels[q];

                if (label_p != label_q) {
                    ext_count++;
                } else {
                    int_count++;
                }
            }

            __sync_fetch_and_add(sizes + label_p, 1);
            if (ext_count > 0) __sync_fetch_and_add(ext_edges + label_p, ext_count);
            if (int_count > 0) __sync_fetch_and_add(int_edges + label_p, int_count);
        }
    }

    timer_end();
}

Partition *create_initial_partition(const Graph *g) {
    timer_start("create initial partition");

    unsigned int n = g->n;
    const unsigned int *indices = g->indices;
    const unsigned int *adj = g->adj;
    const int *tri = g->triangles;

    unsigned int *labels = new unsigned int[n];
    fill(labels, labels + n, INVALID_LABEL);
    unsigned int num_labels = 0;

    pair<pair<double, int>, unsigned int> *order =
        new pair<pair<double, int>, unsigned int>[n];

    for (unsigned int p = 0; p < n; p++) {
        unsigned int deg = g->degree(p);
        double cc = tri[p] / (deg * (deg - 1.0));

        order[p] = make_pair(make_pair(cc, deg), p);
    }

    sort(order, order + n, greater<pair<pair<double, int>, unsigned int> >());

    for (unsigned int i = 0; i < n; i++) {
        unsigned int p = order[i].second;

        if (labels[p] != INVALID_LABEL) {
            continue;
        }

        unsigned int l = num_labels;

        for (unsigned int j = indices[p]; j != indices[p + 1]; j++) {
            unsigned int q = adj[j];

            if (labels[q] == INVALID_LABEL) {
                labels[q] = l;
            }
        }

        labels[p] = l;
        num_labels++;
    }

    delete[] order;

    Partition *par = new Partition;
    par->labels = labels;
    par->num_communities = num_labels;
    par->sizes = new int[n];
    par->ext_edges = new int[n];
    par->int_edges = new int[n];

    timer_end();
    collect_stats(g, par);

    return par;
}

inline static double calculate_improvement(int r, int d_in, int d_out, int c_out,
        double p_in, double p_ext, double alpha) {
    double t;
    if (r > 0) {
        t = (c_out - d_in) / (double) r;
    } else {
        t = 0.0;
    }

    // Node v
    double A = 0.0;
    double denom = 0.0;
    denom = (d_in * (d_in - 1) * p_in +
            d_out * (d_out - 1) * p_ext) +
            d_out * d_in * p_ext;
    denom *= d_in + d_out + alpha*(r-1-d_in);
    if (denom != 0.0) {
        A = ((d_in * (d_in - 1) * p_in) * (d_in + d_out)) / (denom);
    }

    // Nodes connected with v
    double BMinus = 0.0;
    denom = (r - 1)*(r - 2) * p_in * p_in * p_in +
            2*(d_in - 1) * p_in +
            t * (r - 1) * p_in * p_ext +
            t * (t - 1) * p_ext +
            (d_out) * p_ext;
    denom *= (r-1)*p_in + 1 + t + alpha*(r - (r-1)*p_in - 1);
    if (denom != 0.0) {
        BMinus = (2*(d_in - 1) * p_in) * ((r - 1) * p_in + 1 + t) / denom;
    }

    // Nodes not connected with v
    double CMinus = 0.0;
    denom = (r - 1)*(r - 2) * p_in * p_in * p_in +
            t * (t - 1) * p_ext +
            t * (r - 1)*(p_in) * p_ext;
    denom *= (r-1)*p_in + t + alpha*(r - (r-1)*p_in);
    denom *= (r-1)*p_in + t + alpha*(r - (r-1)*p_in - 1);
    if (denom != 0.0 && ((r + t) > 0) && ((r - 1 + t) > 0)) {
        CMinus = -((r - 1)*(r - 2) * p_in * p_in * p_in) * ((r - 1) * p_in
                + t)*alpha / denom;
    }

    // Total
    return (A + d_in * BMinus + (r - d_in) * CMinus);
}

void improve_partition(const Graph *g, Partition *par, double alpha,
        const bool *mask) {
    timer_start("improving partition");

    unsigned int n = g->n;
    const unsigned int *const adj = g->adj;
    const unsigned int *const indices = g->indices;

    unsigned int *new_labels = new unsigned int[n];
    const unsigned int *old_labels = par->labels;
    const int *sizes = par->sizes;
    const int *ext_edges = par->ext_edges;
    const int *int_edges = par->int_edges;

#pragma omp parallel
    {
        unsigned int buffer_cap = 4096;
        unsigned int *buffer = new unsigned int[buffer_cap];

#pragma omp for schedule(dynamic,512)
        for (unsigned int p = 0; p < n; p++) {
            if (mask != NULL && !mask[p]) {
                continue;
            }

            unsigned int deg = g->degree(p);
            unsigned int my_label = old_labels[p];

            if (deg > buffer_cap) {
                delete[] buffer;
                buffer = new unsigned int[deg];
                buffer_cap = deg;
            }

            for (unsigned int i = 0; i < deg; i++) {
                buffer[i] = old_labels[adj[indices[p] + i]];
            }

            sort(buffer, buffer + deg);

            double remove_improvement = 0.0;
            double insert_improvement = 0.0;
            unsigned int insert_label = 0;

            unsigned int index = 0;
            while (index < deg) {
                unsigned int other_label = buffer[index];
                unsigned int freq = 1;

                if (other_label == INVALID_LABEL) {
                    index++;
                    continue;
                }

                while(index + freq < deg && buffer[index + freq] == other_label) {
                    freq++;
                }


                int size = sizes[other_label];
                int ext_edge = ext_edges[other_label];
                int int_edge = int_edges[other_label] / 2;

                if (other_label == my_label) {
                    size--;
                    ext_edge += (int) (freq - (deg - freq));
                    int_edge -= (int) freq;
                }

                int d_in = (int) freq;
                int d_out = (int) deg - (int) d_in;
                int c_out = ext_edge;
                double p_int = 0.0;
                double p_ext = g->clustering_coef;

                if (size > 1) {
                    p_int = (2.0 * int_edge) / (size * (size - 1.0));
                }

                double improvement = calculate_improvement(
                    size,
                    d_in,
                    d_out,
                    c_out,
                    p_int,
                    p_ext,
                    alpha);

                if (other_label == my_label) {
                    remove_improvement = -improvement;
                }

                if (improvement > insert_improvement) {
                    insert_improvement = improvement;
                    insert_label = other_label;
                }

                index += freq;
            }

            unsigned int new_label;
            insert_improvement += remove_improvement;

            if (insert_improvement > remove_improvement && insert_improvement > 0.0) {
                new_label = insert_label;
            } else if (remove_improvement > 0) {
                new_label = INVALID_LABEL;
            } else {
                new_label = my_label;
            }

            new_labels[p] = new_label;
        }

        delete[] buffer;
    }

    par->labels = new_labels;
    delete[] old_labels;
    timer_end();
}
