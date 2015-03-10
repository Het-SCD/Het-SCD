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
#include <bitset>
#include <cuda.h>
#include <iostream>
#include <moderngpu.cuh>
#include <omp.h>

#include "scd.hpp"
#include "wcc.hpp"
#include "cuda.hpp"
#include "cuda-common.cuh"
#include "kernels.cuh"

#define MAX_DEVICES (15)
#define MAX_PARTITIONS (MAX_DEVICES + 1)

using namespace std;

static int num_devices;
static mgpu::ContextPtr contexts[MAX_DEVICES];

static const Graph *graph;
static Partition *comms;
static int block_size;
static bool profiling;

static int num_host_vertices;
static int num_host_total_out_ghosts;
static int num_host_total_in_ghosts;
static int num_host_out_ghosts[MAX_DEVICES];
static int num_host_in_ghosts[MAX_DEVICES];
static int *host_out_ghosts;
static int *host_in_ghosts;
static bool *host_vertices_mask;

static int *host_in_ghost_labels;
static int *host_out_ghost_labels;

static int num_dev_vertices[MAX_DEVICES];
static int num_dev_total_vertices[MAX_DEVICES];
static int num_dev_ghosts[MAX_DEVICES][MAX_DEVICES];
static int num_dev_in_ghosts[MAX_DEVICES];
static int num_dev_out_ghosts[MAX_DEVICES];
static int num_dev_end_points[MAX_DEVICES];
static int num_dev_total_end_points[MAX_DEVICES];

static int *dev_vertices[MAX_DEVICES];

static MGPU_MEM(int) mem_sources[MAX_DEVICES];
static MGPU_MEM(int) mem_adj[MAX_DEVICES];
static MGPU_MEM(int) mem_indices[MAX_DEVICES];

static MGPU_MEM(int) mem_triangles[MAX_DEVICES];
static MGPU_MEM(int) mem_triangle_degs[MAX_DEVICES];
static MGPU_MEM(int) mem_internal_triangles[MAX_DEVICES];
static MGPU_MEM(int) mem_internal_triangle_degs[MAX_DEVICES];

static MGPU_MEM(int) mem_comm_sizes[MAX_DEVICES];
static MGPU_MEM(int) mem_comm_int[MAX_DEVICES];
static MGPU_MEM(int) mem_comm_ext[MAX_DEVICES];

static int *dev_comm_sizes[MAX_DEVICES];
static int *dev_comm_int[MAX_DEVICES];
static int *dev_comm_ext[MAX_DEVICES];

static MGPU_MEM(int) mem_labels[MAX_DEVICES];
static MGPU_MEM(int) mem_best_labels[MAX_DEVICES];
static MGPU_MEM(double) mem_wcc[MAX_DEVICES];

static MGPU_MEM(int) mem_out_ghost_vertices[MAX_DEVICES];
static MGPU_MEM(int) mem_out_ghost_labels[MAX_DEVICES];
static MGPU_MEM(int) mem_in_ghost_labels[MAX_DEVICES];


bool dist_init(vector<int> device_ids, int work_group_size, bool profile_flag) {
    int count = mgpu::CudaDevice::DeviceCount();
    bool err = device_ids.empty();

    for (unsigned int i = 0; i < device_ids.size(); i++) {
        int id = device_ids[i];

        if (id < 0 || id >= count) {
            err = true;
            break;
        }

        contexts[i] = mgpu::CreateCudaDeviceStream(id);
    }

    if (err) {
        cerr << "Please select a CUDA-enabled device:" << endl;
        cerr << endl;

        for (int i = 0; i < count; i++) {
            mgpu::CudaDevice &d = mgpu::CudaDevice::ByOrdinal(i);

            cerr << d.DeviceString();
        }

        if (count == 0) {
            cerr << "No devices found!" << endl;
        }

        return false;
    }

    num_devices = device_ids.size();
    block_size = work_group_size;
    profiling = profile_flag;

    cout << "selected devices:" << endl;

    for (int i = 0; i < num_devices; i++) {
        cout << contexts[i]->DeviceString();
    }


    return true;
}

template <size_t S>
void read_neighboring_partitions(const Graph *g, int *partition, bitset<S> *neighbors) {
    const int n = (int) g->n;
    const int *const indices = (int*) g->indices;
    const int *const adj = (int*) g->adj;

    for (int i = 0; i < n; i++) {
        bitset<S> s;
        s.set(partition[i]);

        for (int j = indices[i]; j < indices[i + 1]; j++) {
            s.set(partition[adj[j]]);
        }

        neighbors[i] = s;
    }
}

template <size_t S>
void extract_ghosts(int source, int target, const Graph *g, int *partition,
        bitset<S> *neighbors, vector<int> &ghosts) {
    const int n = (int) g->n;

    for (int i = 0; i < n; i++) {
        if (partition[i] == source && neighbors[i][target]) {
            ghosts.push_back(i);
        }
    }
}

template <size_t S>
void extract_partition(int id, const Graph *g, int *partition, bitset<S> *neighbors,
        int &pn, int &ps, int *&pindices, int *&padj, int *&local2global,
        int *&global2local) {
    const int n = (int) g->n;
    const int *const indices = (int*) g->indices;
    const int *const adj = (int*) g->adj;
    const int num_partitions = num_devices + 1;

    ps = 0;
    pn = 0;

    for (int i = 0; i < n; i++) {
        if (partition[i] == id || neighbors[i][id]) {
            ps++;
        }

        if (partition[i] == id) {
            pn++;
        }
    }

    int index = 0;
    local2global = new int[ps];

    for (int i = 0; i < n; i++) {
        if (partition[i] == id) {
            local2global[index++] = i;
        }
    }

    for (int i = 0; i < num_partitions; i++) {
        vector<int> ghosts;

        if (i != id) {
            extract_ghosts(i, id, g, partition, neighbors, ghosts);
        }

        copy(ghosts.begin(), ghosts.end(), local2global + index);
        index += ghosts.size();
    }

    int deg_sum = 0;
    for (int i = 0; i < ps; i++) {
        deg_sum += g->degree(local2global[i]);
    }

    padj = new int[deg_sum];
    pindices = new int[ps + 1];

    global2local = new int[n];
    fill(global2local, global2local + n, -1);

    for (int i = 0; i < ps; i++) {
        global2local[local2global[i]] = i;
    }

    index = 0;
    pindices[0] = 0;

    for (int i = 0; i < ps; i++) {
        int p = local2global[i];

        for (int j = indices[p]; j != indices[p + 1]; j++) {
            int q = adj[j];

            if (global2local[q] != -1) {
                padj[index++] = global2local[q];
            }
        }

        pindices[i + 1] = index;
    }

    for (int i = 0; i < ps; i++) {
        sort(padj + pindices[i], padj + pindices[i + 1]);
    }
}

template <size_t S>
bool init_host(const Graph *g, int *partition, bitset<S> *neighbors) {
    const int n = (int) g->n;

    host_vertices_mask = new bool[n];
    num_host_vertices = 0;

    for (int i = 0; i < n; i++) {
        host_vertices_mask[i] = (partition[i] == 0);
        num_host_vertices += (partition[i] == 0);
    }

    vector<int> ghosts;

    for (int i = 0; i < num_devices; i++) {
        int s = ghosts.size();
        extract_ghosts(0, i + 1, g, partition, neighbors, ghosts);
        num_host_out_ghosts[i] = ghosts.size() - s;
    }

    num_host_total_out_ghosts = ghosts.size();

    host_out_ghosts = new int[ghosts.size()];
    copy(ghosts.begin(), ghosts.end(), host_out_ghosts);

    CUDA_CALL(cudaHostAlloc, &host_out_ghost_labels,
            sizeof(int) * ghosts.size(), cudaHostAllocPortable);

    ghosts.clear();

    for (int i = 0; i < num_devices; i++) {
        int s = ghosts.size();
        extract_ghosts(i + 1, 0, g, partition, neighbors, ghosts);
        num_host_in_ghosts[i] = ghosts.size() - s;
    }

    num_host_total_in_ghosts = ghosts.size();

    host_in_ghosts = new int[ghosts.size()];
    copy(ghosts.begin(), ghosts.end(), host_in_ghosts);

    CUDA_CALL(cudaHostAlloc, &host_in_ghost_labels,
            sizeof(int) * ghosts.size(), cudaHostAllocPortable);

    cout << "partition host" << endl;
    cout << " - number of vertices: " << num_host_vertices << endl;
    cout << " - number of ghosts: " << endl;

    for (int dev = 0; dev < num_devices; dev++) {
        cout << "    - to device " << dev + 1 << ": " << num_host_out_ghosts[dev] << endl;
    }

    return true;
}

template <size_t S>
bool init_device(int id, const Graph *g, int *partition, bitset<S> *neighbors) {
    const int n = (int) g->n;
    const int *const triangles = (int*) g->triangles;
    const int *const triangle_deg = (int*) g->triangle_deg;

    int pn, ps;
    int *padj, *pindices, *map, *rmap;

    extract_partition(id + 1, g, partition, neighbors,
            pn, ps, pindices, padj, map, rmap);

    int *psources = new int[pindices[ps]];

    for (int i = 0; i < ps; i++) {
        for (int j = pindices[i]; j < pindices[i + 1]; j++) {
            psources[j] = i;
        }
    }

    vector<int> ghosts;
    extract_ghosts(id + 1, 0, g, partition, neighbors, ghosts);
    num_host_in_ghosts[id] = ghosts.size();

    for (int i = 0; i < num_devices; i++) {
        int s = ghosts.size();

        if (i != id) {
            extract_ghosts(id + 1, i + 1, g, partition, neighbors, ghosts);
        }

        num_dev_ghosts[id][i] = ghosts.size() - s;
    }

    for (int i = 0; i < ghosts.size(); i++) {
        ghosts[i] = rmap[ghosts[i]];
    }

    int *tri = new int[pn];
    int *trid = new int[pn];

    for (int i = 0; i < pn; i++) {
        tri[i] = triangles[map[i]];
        trid[i] = triangle_deg[map[i]];
    }

    mgpu::ContextPtr ctx = contexts[id];
    ctx->SetActive();

    size_t before_free, after_free, total;
    CUDA_CALL(cudaMemGetInfo, &before_free, &total);

    mem_indices[id] = ctx->Malloc(pindices, ps + 1);
    mem_sources[id] = ctx->Malloc(psources, pindices[ps]);
    mem_adj[id] = ctx->Malloc(padj, pindices[ps]);

    mem_triangles[id] = ctx->Malloc<int>(tri, pn);
    mem_triangle_degs[id] = ctx->Malloc<int>(trid, pn);
    mem_internal_triangles[id] = ctx->Malloc<int>(pn);
    mem_internal_triangle_degs[id] = ctx->Malloc<int>(pn);

    mem_comm_sizes[id] = ctx->Malloc<int>(n);
    mem_comm_int[id] = ctx->Malloc<int>(n);
    mem_comm_ext[id] = ctx->Malloc<int>(n);

    mem_out_ghost_vertices[id] = ctx->Malloc(ghosts);
    mem_out_ghost_labels[id] = ctx->Malloc<int>(ghosts.size());
    mem_in_ghost_labels[id] = ctx->Malloc<int>(ps - pn);

    mem_wcc[id] = ctx->Malloc<double>(n);

    num_dev_vertices[id] = pn;
    num_dev_total_vertices[id] = ps;
    num_dev_end_points[id] = pindices[pn];
    num_dev_total_end_points[id] = pindices[ps];
    num_dev_in_ghosts[id] = ps - pn;
    num_dev_out_ghosts[id] = ghosts.size();
    dev_vertices[id] = map;

    delete[] pindices;
    delete[] padj;
    delete[] psources;
    delete[] rmap;
    delete[] tri;
    delete[] trid;

    CUDA_CALL(cudaMemGetInfo, &after_free, &total);

    size_t reserved = total - before_free;
    size_t used = before_free - after_free;
    size_t avail = after_free;

    cout << "device #" << id + 1 << ": " << endl;
    cout << " - memory:" << endl;
    cout << "    - " << reserved << " bytes reserved ("
        << reserved * 100 / total << "%)" << endl;
    cout << "    - " << used << " bytes used ("
        << used * 100 / total << "%)" << endl;
    cout << "    - " << avail << " bytes available ("
        << avail * 100 / total << "%)" << endl;

    cout << " - number of vertices: " << num_dev_vertices[id] << endl;
    cout << " - number of edges: " << num_dev_total_end_points[id] / 2 << endl;
    cout << " - number of ghosts: " << endl;

    cout << "    - to host : " << num_host_in_ghosts[id] << endl;

    for (int i = 0; i < num_devices; i++) {
        if (i != id) {
            cout << "    - to device " << i << ": "
                << num_dev_ghosts[id][i] << endl;
        }
    }

    return true;
}

bool dist_transfer_graph(const Graph *g, int *partition, double alpha) {
    bitset<MAX_DEVICES + 1> *neighbors = new bitset<MAX_DEVICES + 1>[g->n];
    read_neighboring_partitions(g, partition, neighbors);

    init_host(g, partition, neighbors);

    for (int i = 0; i < num_devices; i++) {
        init_device(i, g, partition, neighbors);
    }

    CUDA_SYNC();
    graph = g;

    return true;
}


void dist_labels_to_devices(Partition *prt) {
    comms = prt;

    int num_comms = (int) comms->num_communities;
    int *labels = (int*) comms->labels;
    int *comm_sizes = comms->sizes;
    int *comm_int = comms->int_edges;
    int *comm_ext = comms->ext_edges;

    for (int i = 0; i < num_devices; i++) {
        int pn = num_dev_vertices[i];
        int pg = num_dev_in_ghosts[i];
        int ps = pn + pg;
        int *plabels = new int[ps];

        for (int p = 0; p < ps; p++) {
            plabels[p] = labels[dev_vertices[i][p]];
        }

        mgpu::ContextPtr ctx = contexts[i];
        ctx->SetActive();

        mem_labels[i] = ctx->Malloc(plabels, ps);
        mem_best_labels[i] = ctx->Malloc(plabels, ps);
        mem_comm_sizes[i] = ctx->Malloc(comm_sizes, num_comms);
        mem_comm_int[i] = ctx->Malloc(comm_int, num_comms);
        mem_comm_ext[i] = ctx->Malloc(comm_ext, num_comms);

        size_t s = sizeof(int) * num_comms;
        CUDA_CALL(cudaHostAlloc, (void**) &dev_comm_sizes[i], s, cudaHostAllocPortable);
        CUDA_CALL(cudaHostAlloc, (void**) &dev_comm_int[i], s, cudaHostAllocPortable);
        CUDA_CALL(cudaHostAlloc, (void**) &dev_comm_ext[i], s, cudaHostAllocPortable);

        CUDA_SYNC();
        delete[] plabels;
    }
}

void dist_improve_labels(double *wcc) {
    int num_comms = (int) comms->num_communities;
    int *comm_sizes = comms->sizes;
    int *comm_int = comms->int_edges;
    int *comm_ext = comms->ext_edges;
    int *labels = (int*) comms->labels;

    double result = 0.0;

    omp_set_nested(1);

#pragma omp parallel num_threads(num_devices + 1) reduction(+:result)
    {
        int tid = omp_get_thread_num();
        int i, n, m, tn, tm;
        mgpu::ContextPtr ctx;
        string name;

        if (tid == 0) {
            n = num_host_vertices;
            name = "host";
        } else {
            i = tid - 1;
            name = "device #" + to_string(i);
            n = num_dev_vertices[i];
            m = num_dev_end_points[i];
            tn = num_dev_total_vertices[i];
            tm = num_dev_total_end_points[i];

            ctx = contexts[i];
            ctx->SetActive();
        }

#pragma omp master
        {
            timer_start("all improve partition");
        }

        if (tid == 0) {
            improve_partition(graph, comms, 1.0, host_vertices_mask);
            labels = (int*) comms->labels;

            for (int i = 0; i < num_host_total_out_ghosts; i++) {
                host_out_ghost_labels[i] = labels[host_out_ghosts[i]];
            }

        } else {
            MGPU_MEM(int) mem_adj_labels = ctx->Malloc<int>(m);
            MGPU_MEM(int2) mem_pairs = ctx->Malloc<int2>(m);
            MGPU_MEM(int) mem_pairs_count = ctx->Malloc<int>(m);
            MGPU_MEM(int) mem_pairs_sources = mem_adj_labels;
            MGPU_MEM(int2) mem_pairs_improv = mem_pairs;

            MGPU_MEM(int2) mem_vertex_insert_improv = ctx->Malloc<int2>(n);
            MGPU_MEM(int) mem_vertex_remove_improv = ctx->Malloc<int>(n);

            CUDA_LAUNCH(ctx, m, set_adj_labels,
                    m,
                    mem_labels[i],
                    mem_adj[i],
                    mem_adj_labels);

            mgpu::SegSortKeysFromIndices(
                    mem_adj_labels->get(),
                    m,
                    mem_indices[i]->get(),
                    n + 1,
                    *ctx);

            int num_labels;
            mgpu::ReduceByKey(
                    make_int2_zip_iterator(
                        mem_sources[i]->get(),
                        mem_adj_labels->get()
                    ),
                    mgpu::constant_iterator<int>(1),
                    m,
                    0,
                    mgpu::plus<int>(),
                    mgpu::equal_to<int2>(),
                    mem_pairs->get(),
                    mem_pairs_count->get(),
                    &num_labels,
                    (int*) NULL,
                    *ctx);

            CUDA_CLEAR(ctx, mem_vertex_remove_improv, n);

            CUDA_LAUNCH(ctx, num_labels, calculate_label_improvements,
                    num_labels,
                    graph->clustering_coef,
                    mem_pairs,
                    mem_pairs_count,
                    mem_labels[i],
                    mem_indices[i],
                    mem_comm_sizes[i],
                    mem_comm_ext[i],
                    mem_comm_int[i],
                    mem_pairs_sources,
                    mem_pairs_improv,
                    mem_vertex_remove_improv);

            mgpu::ReduceByKey(
                    mem_pairs_sources->get(),
                    mem_pairs_improv->get(),
                    num_labels,
                    make_int2(0, -1),
                    max_second_min_first(),
                    mgpu::equal_to<int>(),
                    (int*) NULL,
                    mem_vertex_insert_improv->get(),
                    (int*) NULL,
                    (int*) NULL,
                    *ctx);

            CUDA_LAUNCH(ctx, n, update_labels,
                    n,
                    mem_vertex_insert_improv,
                    mem_vertex_remove_improv,
                    mem_labels[i]);

            if (num_dev_out_ghosts[i]) {
                CUDA_LAUNCH(ctx, num_dev_out_ghosts[i], prepare_outgoing_ghosts,
                        num_dev_out_ghosts[i],
                        mem_labels[i],
                        mem_out_ghost_vertices[i],
                        mem_out_ghost_labels[i]);
            }

            cudaStreamSynchronize(ctx->Stream());
        }

#pragma omp barrier
#pragma omp master
        {
            timer_end();
            timer_start("all exchange labels");
        }

        if (tid > 0) {
            // host -> device on AuxStream
            if (num_host_out_ghosts[i]) {
                int size = num_host_out_ghosts[i];
                int offset = 0;

                for (int k = 0; k < i; k++) {
                    offset += num_host_out_ghosts[k];
                }

                cudaMemcpyAsync(
                        mem_labels[i]->get() + n,
                        host_out_ghost_labels + offset,
                        size * sizeof(int),
                        cudaMemcpyHostToDevice,
                        ctx->AuxStream());
            }

            int src_offset = 0;

            // device -> host on Stream
            if (num_host_in_ghosts[i]) {
                int size = num_host_in_ghosts[i];
                int dst_offset = 0;

                for (int k = 0; k < i; k++) {
                    dst_offset += num_host_in_ghosts[k];
                }

                CUDA_CALL(cudaMemcpyAsync,
                        host_in_ghost_labels + dst_offset,
                        mem_out_ghost_labels[i]->get() + src_offset,
                        size * sizeof(int),
                        cudaMemcpyDeviceToHost,
                        ctx->Stream());

                vector<int> bla;
                mem_out_ghost_labels[i]->ToHost(bla, size);

                src_offset += size;
            }

            // device -> device on Stream
            for (int j = 0; j < num_devices; j++) {
                int size = num_dev_ghosts[i][j];

                if (size > 0) {
                    int dst_offset = num_dev_vertices[j] + num_host_out_ghosts[j];

                    for (int k = 0; k < i; k++) {
                        dst_offset += num_dev_ghosts[k][j];
                    }

                    int dev_i = contexts[i]->Device().Ordinal();
                    int dev_j = contexts[j]->Device().Ordinal();

                    CUDA_CALL(cudaMemcpyPeerAsync,
                            mem_labels[j]->get() + dst_offset,
                            dev_j,
                            mem_out_ghost_labels[i]->get() + src_offset,
                            dev_i,
                            size * sizeof(int),
                            ctx->Stream());

                    src_offset += size;
                }
            }

            // Sync all streams
            CUDA_CALL(cudaStreamSynchronize, ctx->AuxStream());
            CUDA_CALL(cudaStreamSynchronize, ctx->Stream());
        }

#pragma omp barrier
#pragma omp master
        {
            timer_end();
            timer_start("all collect community statistics");
        }

        if (tid == 0) {
            for (int i = 0; i < num_host_total_in_ghosts; i++) {
                labels[host_in_ghosts[i]] = host_in_ghost_labels[i];
            }

            collect_stats(graph, comms, host_vertices_mask);
        } else {
            CUDA_FILL(ctx, mem_comm_sizes[i], num_comms, 0);
            CUDA_FILL(ctx, mem_comm_int[i], num_comms, 0);
            CUDA_FILL(ctx, mem_comm_ext[i], num_comms, 0);

            CUDA_LAUNCH(ctx, n, update_comm_sizes,
                    n,
                    mem_labels[i],
                    mem_comm_sizes[i]);

            CUDA_LAUNCH(ctx, m, update_comm_edges,
                    m,
                    mem_labels[i],
                    mem_sources[i],
                    mem_adj[i],
                    mem_comm_int[i],
                    mem_comm_ext[i]);

            CUDA_CALL(cudaMemcpyAsync,
                    dev_comm_sizes[i],
                    mem_comm_sizes[i]->get(),
                    num_comms * sizeof(int),
                    cudaMemcpyDeviceToHost,
                    ctx->Stream());

            CUDA_CALL(cudaMemcpyAsync,
                    dev_comm_ext[i],
                    mem_comm_ext[i]->get(),
                    num_comms * sizeof(int),
                    cudaMemcpyDeviceToHost,
                    ctx->Stream());

            CUDA_CALL(cudaMemcpyAsync,
                    dev_comm_int[i],
                    mem_comm_int[i]->get(),
                    num_comms * sizeof(int),
                    cudaMemcpyDeviceToHost,
                    ctx->Stream());

            CUDA_CALL(cudaStreamSynchronize, ctx->Stream());
        }

#pragma omp barrier
#pragma omp master
        {
            timer_end();
            timer_start("reduce community statistics");
        }

        if (tid == 0) {
            for (int i = 0; i < num_devices; i++) {

#pragma omp parallel for
                for (int j = 0; j < num_comms; j++) {
                    comm_sizes[j] += dev_comm_sizes[i][j];
                    comm_ext[j] += dev_comm_ext[i][j];
                    comm_int[j] += dev_comm_int[i][j];
                }
            }
        }

#pragma omp barrier
#pragma omp master
        {
            timer_end();
            timer_start("all compute WCC");
        }

        if (tid == 0) {
            result = compute_wcc(graph, comms, 1.0, host_vertices_mask) * num_host_vertices;
        } else {
            CUDA_CALL(cudaMemcpyAsync,
                    mem_comm_sizes[i]->get(),
                    comm_sizes,
                    num_comms * sizeof(int),
                    cudaMemcpyHostToDevice,
                    ctx->Stream());

            CUDA_CALL(cudaMemcpyAsync,
                    mem_comm_ext[i]->get(),
                    comm_ext,
                    num_comms * sizeof(int),
                    cudaMemcpyHostToDevice,
                    ctx->Stream());

            CUDA_CALL(cudaMemcpyAsync,
                    mem_comm_int[i]->get(),
                    comm_int,
                    num_comms * sizeof(int),
                    cudaMemcpyHostToDevice,
                    ctx->Stream());

            CUDA_CALL(cudaStreamSynchronize, ctx->Stream());

            MGPU_MEM(int) mem_adj_indices = ctx->Malloc<int>(tm + 1);

            CUDA_LAUNCH(ctx, tm, set_internal_edges,
                    tm,
                    mem_labels[i],
                    mem_sources[i],
                    mem_adj[i],
                    mem_adj_indices);

            int new_tm;
            mgpu::Scan<mgpu::MgpuScanTypeExc>(
                    mem_adj_indices->get(),
                    tm,
                    0,
                    mgpu::plus<int>(),
                    mem_adj_indices->get() + tm,
                    &new_tm,
                    mem_adj_indices->get(),
                    *ctx);

            MGPU_MEM(int) mem_new_indices = ctx->Malloc<int>(tn + 1);
            MGPU_MEM(int) mem_new_sources = ctx->Malloc<int>(new_tm);
            MGPU_MEM(int) mem_new_adj = ctx->Malloc<int>(new_tm);

            CUDA_LAUNCH(ctx, tn + 1, create_indices,
                    tn,
                    mem_adj_indices,
                    mem_indices[i],
                    mem_new_indices);

            CUDA_LAUNCH(ctx, tm, create_adj,
                    tm,
                    mem_adj_indices,
                    mem_sources[i],
                    mem_adj[i],
                    mem_new_sources,
                    mem_new_adj);

            CUDA_FILL(ctx, mem_internal_triangles[i], n, 0);
            CUDA_FILL(ctx, mem_internal_triangle_degs[i], n, 0);

            CUDA_LAUNCH(ctx, new_tm, count_triangles,
                    new_tm,
                    n,
                    mem_labels[i],
                    mem_new_sources,
                    mem_new_adj,
                    mem_new_indices,
                    mem_internal_triangles[i],
                    mem_internal_triangle_degs[i]);

            CUDA_LAUNCH(ctx, n, compute_wccs,
                    n,
                    mem_triangles[i],
                    mem_triangle_degs[i],
                    mem_internal_triangles[i],
                    mem_internal_triangle_degs[i],
                    mem_labels[i],
                    mem_comm_sizes[i],
                    mem_wcc[i]);

            result = mgpu::Reduce(
                    mem_wcc[i]->get(),
                    n,
                    *ctx);


        }

#pragma omp barrier
#pragma omp master
        {
            timer_end();
        }
    }

    *wcc = double(result) / graph->n;
}

void dist_store_labels() {
    for (int i = 0; i < num_devices; i++) {
        mem_best_labels[i]->FromDevice(mem_labels[i]->get(), num_dev_vertices[i]);
    }

    CUDA_SYNC();
}

void dist_labels_from_devices(unsigned int *labels) {

    // copy labels from host
    for (int i = 0; i < graph->n; i++) {
        if (host_vertices_mask[i]) {
            labels[i] = comms->labels[i];
        }
    }

    // copy labels from devices
    for (int i = 0; i < num_devices; i++) {
        int pn = num_dev_vertices[i];
        int *buffer = new int[pn];
        mem_best_labels[i]->ToHost(buffer, pn);

        for (int j = 0; j < pn; j++) {
            labels[dev_vertices[i][j]] = buffer[j];
        }

        free(buffer);
    }
}
