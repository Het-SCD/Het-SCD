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
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>
#include <vector>

#include "common.hpp"

using namespace std;


template <typename T, typename Compare>
void omp_sort(T begin, T end, Compare comp) {
    int nthreads = omp_get_max_threads();
    size_t n = distance(begin, end);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        size_t a = (n * tid) / nthreads;
        size_t b = (n * (tid + 1)) / nthreads;

        sort(begin + a, begin + b, comp);
    }

    for (size_t seg = 1; seg < nthreads; seg *= 2) {
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            size_t a = min((n * (2 * tid + 0) * seg) / nthreads, n);
            size_t b = min((n * (2 * tid + 1) * seg) / nthreads, n);
            size_t c = min((n * (2 * tid + 2) * seg) / nthreads, n);

            if (b != c) {
                inplace_merge(
                        begin + a,
                        begin + b,
                        begin + c,
                        comp
                );
            }
        }
    }
}

template <typename T>
void omp_sort(T begin, T end) {
    typedef typename iterator_traits<T>::value_type V;

    omp_sort(begin, end, less<V>());
}

void read_edges(char *filename, unsigned int &num_nodes, vector<pair<unsigned int, unsigned int> > &edges) {
    ifstream input(filename);
    char line[1024];
    num_nodes = 0;

    // read input
    for (int i = 1; input.good(); i++) {
        int idx = 0;
        input.getline(line, sizeof(line));

        while (isspace(line[idx])) {
            idx++;
        }

        if (line[idx] == '\0' || line[idx] == '#') {
            continue;
        }

        if (!isdigit(line[idx])) {
            cerr << "line " << i << " is corrupt" << endl;
            exit(-1);
        }

        unsigned int a = 0;

        while (isdigit(line[idx])) {
            a = 10 * a + (unsigned int)(line[idx] - '0');
            idx++;
        }

        while (isspace(line[idx])) {
            idx++;
        }

        if (!isdigit(line[idx])) {
            cerr << "line " << i << " is corrupt" << endl;
            exit(-1);
        }

        unsigned int b = 0;

        while (isdigit(line[idx])) {
            b = 10 * b + (unsigned int)(line[idx] - '0');
            idx++;
        }

        edges.push_back(make_pair(
            min(a, b),
            max(a, b)
        ));

        num_nodes = max(max(a, b) + 1, num_nodes);

        if (edges.size() % 1000000 == 0) {
            cout << edges.size() << " lines read so far" << endl;
        }
    }

    if (!input.eof() && input.fail()) {
        cerr << "an error occured while reading input file" << endl;
        exit(-1);
    }

    input.close();
}


void reindex_vertices(unsigned int &num_nodes, vector<pair<unsigned int, unsigned int> > &edges) {
    unsigned int *convert = new unsigned int[num_nodes];
    unsigned int new_num_nodes = 0;
    size_t num_edges = edges.size();

    for (size_t i = 0; i < num_nodes; i++) {
        convert[i] = UINT_MAX;
    }

    for (size_t i = 0; i < num_edges; i++) {
        unsigned int a = edges[i].first;
        unsigned int b = edges[i].second;

        if (convert[a] == UINT_MAX) {
            convert[a] = new_num_nodes++;
        }

        if (convert[b] == UINT_MAX) {
            convert[b] = new_num_nodes++;
        }

        edges[i] = make_pair(
            min(convert[a], convert[b]),
            max(convert[a], convert[b])
        );
    }

    delete[] convert;
    num_nodes = new_num_nodes;
}


void select_largest_component(unsigned int &num_nodes, vector<pair<unsigned int, unsigned int> > &edges) {
    cout << "finding connected components" << endl;

    size_t num_edges = edges.size();

    // find components
    unsigned int *labels = new unsigned int[num_nodes];
    for (unsigned int i = 0; i < num_nodes; i++) {
        labels[i] = i;
    }

    bool changed = true;
    for (int round = 0; changed; round++) {
        changed = false;

        for (size_t i = 0; i < num_edges; i++) {
            unsigned int a = edges[i].first;
            unsigned int b = edges[i].second;

            if (labels[a] != labels[b]) {
                labels[a] = labels[b] = min(labels[a], labels[b]);
                changed = true;
            }
        }

        cout << "finished round " << (round + 1) << endl;
    }


    // find labels used
    unsigned int *labels_count = new unsigned int[num_nodes];
    for (unsigned int i = 0; i < num_nodes; i++) {
        labels_count[i] = 0;
    }

    for (unsigned int i = 0; i < num_nodes; i++) {
        labels_count[labels[i]]++;
    }


    // find largest connected component
    unsigned int num_components = 0;
    unsigned int largest_component = 0;
    unsigned int largest_count = 0;

    for (unsigned int i = 0; i < num_nodes; i++) {
        if (labels_count[i] > 0) {
            num_components++;

            if (labels_count[i] > largest_count) {
                largest_component = i;
                largest_count = labels_count[i];
            }
        }
    }

    cout << "found " << num_components << " components, largest component has "
        << largest_count << " nodes" << endl;

    if (largest_count == num_nodes) {
        return;
    }

    size_t index = 0;

    for (size_t i = 0; i < num_edges; i++) {
        if (labels[edges[i].first] == largest_component) {
            edges[index++] = edges[i];
        }
    }

    edges.resize(index);
    num_nodes = largest_count;

    delete[] labels_count;
    delete[] labels;
}


void remove_duplicates_and_loops(unsigned int &num_nodes, vector<pair<unsigned int, unsigned int> > &edges) {
    (void) num_nodes; // unused

    size_t index = 0;
    size_t num_edges = edges.size();

    for (size_t i = 0; i < num_edges; i++) {
        unsigned int a = edges[i].first;
        unsigned int b = edges[i].second;

        if (a != b && (i == 0 || edges[i] != edges[i - 1])) {
            edges[index++] = edges[i];
        }
    }

    edges.resize(index);
}


void remove_edges_without_triangles(unsigned int &num_nodes, vector<pair<unsigned int, unsigned int> > &edges,
        int *&tri, int *&tri_deg) {
    unsigned int *indices = new unsigned int[num_nodes + 1];
    size_t num_edges = edges.size();

    size_t index = 0;
    indices[0] = 0;

    for (size_t i = 0; i < num_nodes; i++) {
        while (index < num_edges && edges[index].first == i) {
            index++;
        }

        indices[i + 1] = (unsigned int) index;
    }

    int *t = new int[num_edges];
    fill(t, t + num_edges, 0);

#pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < num_edges; i++) {
        unsigned int p = edges[i].first;
        unsigned int q = edges[i].second;

        unsigned int a = indices[p];
        unsigned int b = indices[q];

        unsigned int a_end = indices[p + 1];
        unsigned int b_end = indices[q + 1];

        while (a < a_end && b < b_end) {
            unsigned int x = edges[a].second;
            unsigned int y = edges[b].second;

            if (x == y) {
                __sync_fetch_and_add(t + i, 1);
                __sync_fetch_and_add(t + a, 1);
                __sync_fetch_and_add(t + b, 1);
            }

            if (x <= y) {
                a++;
            }

            if (x >= y) {
                b++;
            }
        }
    }

    tri = new int[num_nodes];
    tri_deg = new int[num_nodes];

    fill(tri, tri + num_nodes, 0);
    fill(tri_deg, tri_deg + num_nodes, 0);

#pragma omp parallel for schedule(guided)
    for (unsigned int i = 0; i < num_edges; i++) {
        unsigned int p = edges[i].first;
        unsigned int q = edges[i].second;
        int c = t[i];

        if (c > 0) {
            __sync_fetch_and_add(tri_deg + p, 1);
            __sync_fetch_and_add(tri_deg + q, 1);

            __sync_fetch_and_add(tri + p, c);
            __sync_fetch_and_add(tri + q, c);
        }
    }

    index = 0;

    for (unsigned int i = 0; i < num_edges; i++) {
        if (t[i] > 0) {
            edges[index++] = edges[i];
        }
    }

    edges.resize(index);
}


void sort_on_degree(unsigned int &num_nodes,
        vector<pair<unsigned int, unsigned int> > &edges,
        int *&tri, int *&tri_deg) {
    pair<unsigned int, unsigned int> *degree_ids = new pair<unsigned int, unsigned int>[num_nodes];
    size_t num_edges = edges.size();
    unsigned int *map = new unsigned int[num_nodes];
    int *new_tri_deg = new int[num_nodes];
    int *new_tri = new int[num_nodes];

    int *deg = new int[num_nodes];
    fill(deg, deg + num_nodes, 0);

#pragma omp parallel for schedule(guided)
    for (unsigned int i = 0; i < num_edges; i++) {
        unsigned int p = edges[i].first;
        unsigned int q = edges[i].second;

        __sync_fetch_and_add(deg + p, 1);
        __sync_fetch_and_add(deg + q, 1);
    }

#pragma omp parallel for schedule(guided)
    for (unsigned int i = 0; i < num_nodes; i++) {
        degree_ids[i] = make_pair(deg[i], i);
    }

    omp_sort(degree_ids, degree_ids + num_nodes,
            greater<pair<unsigned int, unsigned int> >());

#pragma omp parallel for schedule(guided)
    for (unsigned int i = 0; i < num_nodes; i++) {
        unsigned int old_index = degree_ids[i].second;
        unsigned int new_index = i;

        map[old_index] = new_index;

        new_tri[new_index] = tri[old_index];
        new_tri_deg[new_index] = tri_deg[old_index];
    }

    unsigned int new_num_nodes = 0;

#pragma omp parallel for schedule(guided) reduction(max:new_num_nodes)
    for (unsigned int i = 0; i < num_edges; i++) {
        edges[i].first  = map[edges[i].first];
        edges[i].second = map[edges[i].second];

        new_num_nodes = max(new_num_nodes, edges[i].first + 1);
        new_num_nodes = max(new_num_nodes, edges[i].second + 1);
    }

    delete[] degree_ids;
    delete[] map;
    delete[] tri;
    delete[] tri_deg;
    delete[] deg;

    tri = new_tri;
    tri_deg = new_tri_deg;
    num_nodes = new_num_nodes;
}


void write_edges(char *filename, unsigned int num_nodes,
        vector<pair<unsigned int, unsigned int> > &edges,
        int *triangles, int *triangle_degs) {
    size_t num_edges = edges.size();
    edges.resize(2 * num_edges);

#pragma omp parallel for
    for (size_t i = 0; i < num_edges; i++) {
        edges[num_edges + i] = make_pair(edges[i].second, edges[i].first);
    }

    omp_sort(edges.begin(), edges.end());

    ofstream output(filename, ios::binary);
    output.write((char*) &num_nodes, sizeof(unsigned int));
    output.write((char*) &num_edges, sizeof(unsigned int));

    unsigned int curr_node = 0;
    output.write((char*) &curr_node, sizeof(unsigned int));

    for (unsigned int i = 0; i < 2 * num_edges; i++) {
        while (edges[i].first != curr_node) {
            output.write((char*) &i, sizeof(unsigned int));
            curr_node++;
        }
    }

    while (curr_node != num_nodes) {
        unsigned int k = (unsigned int) (2 * num_edges);
        output.write((char*) &k, sizeof(unsigned int));
        curr_node++;
    }

    for (size_t i = 0; i < 2 * num_edges; i++) {
        output.write((char*) &(edges[i].second), sizeof(int));
    }

    output.write((char*) triangles, sizeof(int) * num_nodes);
    output.write((char*) triangle_degs, sizeof(int) * num_nodes);

    if (output.fail()) {
        cerr << "an error occured while writing output file" << endl;
        exit(-1);
    }

    output.close();
}


int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "usage: " << argv[0] << " <input file> <output file>" << endl;
        return -1;
    }

    char *input_file = argv[1];
    char *output_file = argv[2];
    vector<pair<unsigned int, unsigned int> > edges;
    unsigned int num_nodes;

    // read input file
    timer_start("read input");
    cout << "reading graph from file " << input_file << endl;
    read_edges(input_file, num_nodes, edges);
    cout << "found " << num_nodes << " nodes and " << edges.size()
        << " edges" << endl;
    timer_end();

    timer_start("process graph");

    // reindex vertices
    size_t old = num_nodes;
    timer_start("reindexing vertices");
    reindex_vertices(num_nodes, edges);
    timer_end();
    cout << "removed " << (old - num_nodes) << " nodes" << endl;

    // find largest component
    //select_largest_component(num_nodes, edges);
    //cout << "selected largest component which has " << num_nodes
    //    << " nodes and " << (edges.size() / 2) << " edges" << endl;

    // sort edges
    timer_start("sorting edges");
    omp_sort(edges.begin(), edges.end());
    timer_end();

    // remove duplicates and loops
    old = edges.size();
    timer_start("removing duplicate edges and loops");
    remove_duplicates_and_loops(num_nodes, edges);
    timer_end();
    cout << "removed " << (old - edges.size()) << " edges " << endl;

    // remove edges without triangles
    int *tri, *tri_deg;
    old = edges.size();
    timer_start("removing edges without triangles");
    remove_edges_without_triangles(num_nodes, edges, tri, tri_deg);
    timer_end();
    cout << "removed " << (old - edges.size()) << " edges " << endl;

    // sort on degree
    old = num_nodes;
    timer_start("sort nodes according to degree");
    sort_on_degree(num_nodes, edges, tri, tri_deg);
    timer_end();
    cout << "removed " << (old - num_nodes) << " nodes" << endl;

    timer_end();

    // write to file
    cout << "final graph has " << num_nodes << " nodes and "
        << edges.size() << " edges" << endl;
    cout << "writing final graph to " << string(output_file) << endl;
    timer_start("write output");
    write_edges(output_file, num_nodes, edges, tri, tri_deg);
    timer_end();

    return 0;
}
