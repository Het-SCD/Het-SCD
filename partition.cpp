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
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <metis.h>
#include <string>

#include "graph.hpp"

using namespace std;

int main(int argc, char *argv[]) {
    srand(time(NULL));

    if (argc < 4) {
        cerr << "usage: " << argv[0] << " <input file> <output file> <#partitions> "
            "[weights partitions]" << endl;
        return EXIT_FAILURE;
    }

    char *input_file = argv[1];
    char *output_file = argv[2];
    idx_t nparts = atoi(argv[3]);
    real_t *weights = NULL;


    if (argc > 4) {
        if (4 + nparts > argc) {
            cerr << "insuffient number of weights given" << endl;
            return EXIT_FAILURE;
        }

        weights = new real_t[nparts];
        real_t total = 0.0;

        for (int i = 0; i < nparts; i++) {
            weights[i] = (real_t) atof(argv[4 + i]);
            total += weights[i];
        }

        for (int i = 0; i < nparts; i++) {
            weights[i] /= total;
        }
    }

    cout << "loading graph..." << endl;
    const Graph *g = Graph::read(input_file);
    cout << "done" << endl;

    cout << endl;
    cout << "nodes: " << g->n << endl;
    cout << "edges: " << g->m << endl;
    cout << "num partitions: " << nparts << endl;
    cout << endl;

    idx_t nvtxs = g->n;
    idx_t ncon = 3;
    idx_t *xadj = new idx_t[g->n + 1];
    idx_t *adjncy = new idx_t[2 * g->m];
    idx_t *lpart = new idx_t[nvtxs]();
    idx_t options[METIS_NOPTIONS];
    idx_t *vwgt = new idx_t[3 * nvtxs];
    real_t *tpwgts = NULL;
    idx_t nparts_nonzero = nparts;
    idx_t dummy;
    idx_t *map = NULL;

    copy(g->indices, g->indices + g->n + 1, xadj);
    copy(g->adj, g->adj + (2 * g->m), adjncy);

    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_DBGLVL] = METIS_DBG_TIME
                                 | METIS_DBG_CONTIGINFO
                                 | METIS_DBG_COARSEN
                                 | METIS_DBG_REFINE;
    options[METIS_OPTION_SEED] = rand();

    for (unsigned int i = 0; i < g->n; i++) {
        vwgt[3 * i + 0] = 1;
        vwgt[3 * i + 1] = g->degree(i);
        vwgt[3 * i + 2] = g->triangles[i];
    }

    if (weights != NULL) {
        nparts_nonzero = 0;
        tpwgts = new real_t[3 * nvtxs];
        map = new idx_t[nparts];

        for (int i = 0; i < nparts; i++) {
            if (weights[i] > 0) {
                idx_t j = nparts_nonzero++;

                tpwgts[3 * j + 0] = weights[i];
                tpwgts[3 * j + 1] = weights[i];
                tpwgts[3 * j + 2] = weights[i];
                map[j] = i;
            }
        }
    }

    if (nparts_nonzero > 1) {
        cout << "running metis..." << endl;

        int err = METIS_PartGraphKway(
                &nvtxs,
                &ncon,
                xadj,
                adjncy,
                vwgt,
                NULL, // vsize
                NULL, // adjwgt
                &nparts_nonzero,
                tpwgts,
                NULL, // ubvec
                options, // options
                &dummy, // cut
                lpart); // partition

        cout << "done" << endl;

        if (err != METIS_OK) {
            if (err == METIS_ERROR_INPUT) {
                cerr << "METIS: input error" << endl;
            } else if (err == METIS_ERROR_MEMORY) {
                cerr << "METIS: insufficient memory" << endl;
            } else {
                cerr << "METIS: error" << endl;
            }

            return EXIT_FAILURE;
        }
    }

    if (map != NULL) {
        for (unsigned int p = 0; p < g->n; p++) {
            lpart[p] = map[lpart[p]];
        }
    }

    int *part = new int[g->n];
    copy(lpart, lpart + g->n, part);

    long *pnodes = new long[nparts]();
    long *pdegree = new long[nparts]();
    long *ptdegree = new long[nparts]();
    long *pedges = new long[nparts * nparts]();
    long cut = 0;

    for (unsigned int p = 0; p < g->n; p++) {
        pnodes[part[p]]++;
        pdegree[part[p]] += (int) g->degree(p);
        ptdegree[part[p]] += g->triangles[p];

        for (const unsigned int *i = g->begin_neighbors(p); i != g->end_neighbors(p); i++) {
            unsigned int q = *i;

            pedges[part[p] * nparts + part[q]]++;

            if (part[p] != part[q]) {
                cut++;
            }
        }
    }

    cout << endl;
    cout << "results: " << endl;
    cout << "edge cut: " << (cut / 2) << endl;

    for (int i = 0; i < nparts; i++) {
        cout << " - partition " << (i + 1) << endl;
        cout << "   - nodes: " << pnodes[i] << endl;
        cout << "   - sum degree: " << pdegree[i] << endl;
        cout << "   - sum triangles: " << ptdegree[i] << endl;
        cout << "   - edges:" << endl;

        for (int j = 0; j < nparts; j++) {
            if (i != j) {
                cout << "     - to partition " << (j + 1) <<  ": "
                    << pedges[i * nparts + j] << endl;
            }
        }
    }

    cout << endl;
    cout << "writing partitioning to file" << endl;

    ofstream out;
    out.open(output_file, ios::out | ios::binary);
    out.write((char*) part, sizeof(int) * g->n);
    out.close();

    if (out.fail()) {
        cerr << "error while writing file" << endl;
        return EXIT_FAILURE;
    }

    cout << "done" << endl;

    return EXIT_SUCCESS;
}
