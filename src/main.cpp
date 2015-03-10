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
#include <cstdio>
#include <iostream>
#include <omp.h>

#include "graph.hpp"
#include "wcc.hpp"
#include "scd.hpp"
#include "common.hpp"

using namespace std;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "usage: " << argv[0] << " <filename>" << endl;
        return EXIT_FAILURE;
    }

    double alpha = 1.0;
    double threshold = 0.01;

    timer_start("load graph");
    const Graph *g = Graph::read(argv[1]);
    timer_end();

    Partition *p = create_initial_partition(g);
    double best_wcc = compute_wcc(g, p, alpha);

    unsigned int n = g->n;
    unsigned int *best_labels = new unsigned int[n];
    copy(p->labels, p->labels + n, best_labels);

    cout << endl;
    cout << "nodes: " << g->n << endl;
    cout << "edges: " << g->m << endl;
    cout << "avg. cc: " << g->clustering_coef << endl;
    cout << "initial WCC: " << best_wcc << endl;
    cout << "num. threads: " << omp_get_max_threads() << endl;
    cout << endl;

    double before = timer();
    int iters = 0;

    while (1) {
        iters++;

        timer_start("iteration");
        improve_partition(g, p, alpha);
        collect_stats(g, p);
        double wcc = compute_wcc(g, p, alpha);

        if (wcc > best_wcc) {
            copy(p->labels, p->labels + n, best_labels);
        }
        timer_end();

        cout << "WCC: " << wcc << endl;
        cout << endl;

        // Check if improvement is more than threshold
        if (wcc < best_wcc * (1 + threshold)) {
            break;
        }

        best_wcc = wcc;
    }

    double after = timer();
    unsigned int num_comms;
    compress_labels(g, best_labels, &num_comms);

    cout << "----------------------------------------" << endl;
    cout << "num. communities: " << num_comms << endl;
    cout << "WCC: " << best_wcc << endl;
    cout << "iterations: " << iters << endl;
    cout << "time per iteration: " << (after - before) / iters << " ms" << endl;
    cout << "----------------------------------------" << endl;
}
