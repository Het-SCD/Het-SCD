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
#include <cuda.h>
#include <iostream>
#include <moderngpu.cuh>

#include "scd.hpp"
#include "cuda.hpp"
#include "cuda-common.cuh"
#include "kernels.cuh"

using namespace std;

static int num_vertices;
static int num_edges;
static int num_comms;
static double clustering_coef;
static int block_size;
static bool profiling;

mgpu::ContextPtr ctx;

MGPU_MEM(int) mem_sources;
MGPU_MEM(int) mem_adj;
MGPU_MEM(int) mem_indices;

MGPU_MEM(int) mem_triangles;
MGPU_MEM(int) mem_triangle_degs;
MGPU_MEM(int) mem_internal_triangles;
MGPU_MEM(int) mem_internal_triangle_degs;

MGPU_MEM(int) mem_comm_sizes;
MGPU_MEM(int) mem_comm_int;
MGPU_MEM(int) mem_comm_ext;

MGPU_MEM(int) mem_labels;
MGPU_MEM(int) mem_best_labels;


bool cuda_init(int device_id, int work_group_size, bool profile_flag) {
    int count = mgpu::CudaDevice::DeviceCount();

    if (device_id < 0 || device_id >= count) {
        cerr << "Please select a CUDA-enabled device:" << endl;
        cerr << endl;

        for (int i = 0; i < count; i++) {
            mgpu::CudaDevice &d = mgpu::CudaDevice::ByOrdinal(i);

            cerr << d.DeviceString() << endl;
            cerr << endl;
        }

        return false;
    }

    ctx = mgpu::CreateCudaDeviceStream(device_id);
    ctx->SetActive();

    cout << "selected device:" << endl;
    cout << ctx->DeviceString() << endl;

    block_size = work_group_size;
    profiling = profile_flag;

    return true;
}

bool cuda_transfer_graph(const Graph *g, double alpha) {

    // We cast from uint to int since the graph does not fit into the GPU
    // memory anyway if m > 2**31-1
    const int n = (int) g->n;
    const int m = (int) g->m;
    const int *const indices = (int*) g->indices;
    const int *const adj = (int*) g->adj;
    const int *const triangles = g->triangles;
    const int *const triangle_deg = g->triangle_deg;

    int *sources = new int[2 * m];
    int vertex = 0;

    for (int i = 0; i < 2 * m; i++) {
        while (vertex < n && i >= indices[vertex + 1]) {
            vertex++;
        }

        sources[i] = vertex;
    }

    size_t before_free, after_free, total;
    CUDA_CALL(cudaMemGetInfo, &before_free, &total);

    mem_indices = ctx->Malloc(indices, n + 1);
    mem_adj = ctx->Malloc(adj, 2 * m);
    mem_sources = ctx->Malloc(sources, 2 * m);

    mem_triangles = ctx->Malloc<int>(triangles, n);
    mem_triangle_degs = ctx->Malloc<int>(triangle_deg, n);
    mem_internal_triangles = ctx->Malloc<int>(n);
    mem_internal_triangle_degs = ctx->Malloc<int>(n);

    mem_comm_sizes = ctx->Malloc<int>(n);
    mem_comm_int = ctx->Malloc<int>(n);
    mem_comm_ext = ctx->Malloc<int>(n);

    mem_labels = ctx->Malloc<int>(n);
    mem_best_labels = ctx->Malloc<int>(n);

    num_vertices = n;
    num_edges = m;
    clustering_coef = g->clustering_coef;

    CUDA_SYNC();

    delete[] sources;

    CUDA_CALL(cudaMemGetInfo, &after_free, &total);

    size_t reserved = total - before_free;
    size_t used = before_free - after_free;
    size_t avail = after_free;

    cout << "memory:" << endl;
    cout << " - " << reserved << " bytes reserved ("
        << reserved * 100 / total << "%)" << endl;
    cout << " - " << used << " bytes used ("
        << used * 100 / total << "%)" << endl;
    cout << " - " << avail << " bytes available ("
        << avail * 100 / total << "%)" << endl;

    return true;
}

void cuda_labels_to_device(unsigned int *labels) {
    num_comms = *max_element(labels, labels + num_vertices) + 1;

    mem_labels->FromHost((int*) labels, num_vertices);
    mem_best_labels->FromHost((int*) labels, num_vertices);

    CUDA_CLEAR(ctx, mem_comm_sizes->get(), num_comms);
    CUDA_CLEAR(ctx, mem_comm_int->get(), num_comms);
    CUDA_CLEAR(ctx, mem_comm_ext->get(), num_comms);

    CUDA_LAUNCH(ctx, num_vertices, update_comm_sizes,
            num_vertices,
            mem_labels->get(),
            mem_comm_sizes->get());

    CUDA_LAUNCH(ctx, 2 * num_edges, update_comm_edges,
            2 * num_edges,
            mem_labels->get(),
            mem_sources->get(),
            mem_adj->get(),
            mem_comm_int->get(),
            mem_comm_ext->get());

    CUDA_SYNC();
}

void cuda_improve_labels(double *wcc) {
    const int n = num_vertices;
    const int m = 2 * num_edges;

    mgpu::CudaEvent evt0, evt1, evt2, evt3, evt4;

    MGPU_MEM(int) mem_adj_labels = ctx->Malloc<int>(m);
    MGPU_MEM(int2) mem_pairs = ctx->Malloc<int2>(m);
    MGPU_MEM(int) mem_pairs_count = ctx->Malloc<int>(m);
    MGPU_MEM(int) mem_pairs_sources = mem_adj_labels;
    MGPU_MEM(int2) mem_pairs_improv = mem_pairs;

    MGPU_MEM(int2) mem_vertex_insert_improv = ctx->Malloc<int2>(n);
    MGPU_MEM(int) mem_vertex_remove_improv = ctx->Malloc<int>(n);

    MGPU_MEM(int) mem_adj_indices = *reinterpret_cast<MGPU_MEM(int)*>(&mem_pairs);
    MGPU_MEM(int) mem_new_indices = *reinterpret_cast<MGPU_MEM(int)*>(&mem_vertex_insert_improv);
    MGPU_MEM(int) mem_new_sources = mem_adj_labels;
    MGPU_MEM(int) mem_new_adj = mem_pairs_count;
    MGPU_MEM(double) mem_wcc = ctx->Malloc<double>(n);

    CUDA_CALL(cudaEventRecord, (cudaEvent_t) evt0);

    CUDA_LAUNCH(ctx, m, set_adj_labels,
            m,
            mem_labels,
            mem_adj,
            mem_adj_labels);

    mgpu::SegSortKeysFromIndices(
            mem_adj_labels->get(),
            m,
            mem_indices->get(),
            n + 1,
            *ctx);

    int num_labels;
    mgpu::ReduceByKey(
            make_int2_zip_iterator(
                mem_sources->get(),
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
            clustering_coef,
            mem_pairs,
            mem_pairs_count,
            mem_labels,
            mem_indices,
            mem_comm_sizes,
            mem_comm_ext,
            mem_comm_int,
            mem_pairs_sources,
            mem_pairs_improv,
            mem_vertex_remove_improv);

    mgpu::ReduceByKey(
            mem_pairs_sources->get(),
            mem_pairs_improv->get(),
            num_labels,
            make_int2(0, 0),
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
            mem_labels);

    CUDA_CALL(cudaEventRecord, (cudaEvent_t) evt1);

    CUDA_CLEAR(ctx, mem_comm_sizes, num_comms);
    CUDA_CLEAR(ctx, mem_comm_int, num_comms);
    CUDA_CLEAR(ctx, mem_comm_ext, num_comms);

    CUDA_LAUNCH(ctx, n, update_comm_sizes,
            n,
            mem_labels->get(),
            mem_comm_sizes->get());

    CUDA_LAUNCH(ctx, m, update_comm_edges,
            m,
            mem_labels,
            mem_sources,
            mem_adj,
            mem_comm_int,
            mem_comm_ext);

    CUDA_CALL(cudaEventRecord, (cudaEvent_t) evt2);

    CUDA_LAUNCH(ctx, m, set_internal_edges,
            m,
            mem_labels,
            mem_sources,
            mem_adj,
            mem_adj_indices);

    int new_num_edges;
    mgpu::Scan<mgpu::MgpuScanTypeExc>(
            mem_adj_indices->get(),
            m,
            0,
            mgpu::plus<int>(),
            mem_adj_indices->get() + m,
            &new_num_edges,
            mem_adj_indices->get(),
            *ctx);

    CUDA_LAUNCH(ctx, n + 1, create_indices,
            n,
            mem_adj_indices,
            mem_indices,
            mem_new_indices);

    CUDA_LAUNCH(ctx, m, create_adj,
            m,
            mem_adj_indices,
            mem_sources,
            mem_adj,
            mem_new_sources,
            mem_new_adj);

    CUDA_CALL(cudaEventRecord, (cudaEvent_t) evt3);

    CUDA_CLEAR(ctx, mem_internal_triangles->get(), n);
    CUDA_CLEAR(ctx, mem_internal_triangle_degs->get(), n);

    CUDA_LAUNCH(ctx, new_num_edges, count_triangles,
            new_num_edges,
            num_vertices,
            mem_labels,
            mem_new_sources,
            mem_new_adj,
            mem_new_indices,
            mem_internal_triangles,
            mem_internal_triangle_degs);

    CUDA_LAUNCH(ctx, n, compute_wccs,
            n,
            mem_triangles,
            mem_triangle_degs,
            mem_internal_triangles,
            mem_internal_triangle_degs,
            mem_labels,
            mem_comm_sizes,
            mem_wcc);

    *wcc = mgpu::Reduce(
            mem_wcc->get(),
            n,
            *ctx) / n;

    CUDA_CALL(cudaEventRecord, (cudaEvent_t) evt4);

    CUDA_SYNC();

    float time0, time1, time2, time3;
    CUDA_CALL(cudaEventElapsedTime, &time0, (cudaEvent_t) evt0, (cudaEvent_t) evt1);
    CUDA_CALL(cudaEventElapsedTime, &time1, (cudaEvent_t) evt1, (cudaEvent_t) evt2);
    CUDA_CALL(cudaEventElapsedTime, &time2, (cudaEvent_t) evt2, (cudaEvent_t) evt3);
    CUDA_CALL(cudaEventElapsedTime, &time3, (cudaEvent_t) evt3, (cudaEvent_t) evt4);

    cout << "improve partitioning: " << time0 << " ms" << endl;
    cout << "collect statistics: " << time1 << " ms" << endl;
    cout << "remove external edges: " << time2 << " ms" << endl;
    cout << "compute wcc: " << time3 << " ms" << endl;
}

void cuda_store_labels() {
    mem_best_labels->FromDevice(mem_labels->get(), num_vertices);
}

void cuda_labels_from_device(unsigned int *labels) {
    mem_best_labels->ToHost((int*) labels, num_vertices);
}
