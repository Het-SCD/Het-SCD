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
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "graph.hpp"

using namespace std;

Graph::Graph() :
    n(0),
    m(0),
    clustering_coef(0),
    triangle_deg(NULL),
    triangles(NULL),
    indices(NULL),
    adj(NULL) {
}

Graph::Graph(unsigned int num_nodes, unsigned int num_edges, unsigned int *edge_adj, unsigned int *edge_indices) :
    n(num_nodes),
    m(num_edges),
    clustering_coef(0),
    triangle_deg(NULL),
    triangles(NULL),
    indices(edge_indices),
    adj(edge_adj) {
}

Graph::Graph(unsigned int num_nodes, unsigned int num_edges, unsigned int *edge_adj, unsigned int *edge_indices, int *tri_deg, int *tri) :
    n(num_nodes),
    m(num_edges),
    clustering_coef(0),
    triangle_deg(tri_deg),
    triangles(tri),
    indices(edge_indices),
    adj(edge_adj) {

    double cc = 0.0;

#pragma omp parallel for schedule(static, 16) reduction(+:cc)
    for (unsigned int p = 0; p < n; p++) {
        unsigned int deg = degree(p);

        if (deg > 1) {
            cc += tri[p] / (deg * (deg - 1.0));
        }
    }

    *const_cast<double*>(&clustering_coef) = cc / n;
}


Graph::Graph(const Graph &that) :
    n(that.n),
    m(that.m),
    clustering_coef(that.clustering_coef),
    triangle_deg(NULL),
    triangles(NULL),
    indices(new unsigned int[n + 1]),
    adj(new unsigned int[2 * m]) {

    copy(that.indices, that.indices + (n + 1), const_cast<unsigned int*>(indices));
    copy(that.adj, that.adj + (2 * m), const_cast<unsigned int*>(adj));
}


const Graph *Graph::read(const char *filename) {
    unsigned int n, m;
    unsigned int *adj, *indices;
    int *tri, *tri_deg;

    ifstream f(filename, ios_base::binary | ios_base::in);

    if (!f.is_open()) {
        cerr << "failed to open " << filename << endl;
        return NULL;
    }

    f.read((char*) &n, sizeof(unsigned int));
    f.read((char*) &m, sizeof(unsigned int));

    if (!f.good()) {
        cerr << "error while reading " << filename << endl;
        return NULL;
    }

    indices = new unsigned int[n + 1];
    f.read((char*) indices, sizeof(unsigned int) * (n + 1));

    adj = new unsigned int[2 * m];
    f.read((char*) adj, sizeof(unsigned int) * (2 * m));

    tri = new int[n];
    f.read((char*) tri, sizeof(int) * n);

    tri_deg = new int[n];
    f.read((char*) tri_deg, sizeof(int) * n);

    bool err = !f.good();
    f.ignore();
    err |= !f.eof();
    f.close();

    if (err) {
        delete[] adj;
        delete[] indices;
        delete[] tri;
        delete[] tri_deg;

        cerr << "error while reading " << filename << endl;
        return NULL;
    }

    return new Graph(n, m, adj, indices, tri_deg, tri);
}


Graph::~Graph() {
    delete[] adj;
    delete[] indices;
}
