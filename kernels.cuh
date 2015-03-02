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
#include "cuda-common.cuh"

__device__ int pack_float(float f) {
    return * (int*) &f;
}

__device__ float unpack_float(int i) {
    return *(float*) &i;
}


__global__ void set_adj_labels(
        const int m,
        int const* __restrict__ labels,
        int const* __restrict__ adj,
        int * __restrict__ adj_labels) {
    int i = get_global_id();
    if (i >= m) return;

    adj_labels[i] = labels[adj[i]];
}


inline static __device__
double calculate_improvement(int r,
                             int d_in,
                             int d_out,
                             int c_out,
                             double p_in,
                             double p_ext,
                             double alpha) {
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
        CMinus = -((r - 1)*(r - 2) * p_in * p_in * p_in) * ((r - 1) * p_in + t)*alpha / denom;
    }

    // Total
    return (A + d_in * BMinus + (r - d_in) * CMinus);
}



__global__ void calculate_label_improvements(
        const int num_labels,
        const double clustering_coef,
        int2 const *__restrict__ vertex_label_pairs,
        int const *__restrict__ freqs,
        int const *__restrict__ labels,
        int const *__restrict__ indices,
        int const* __restrict__ comm_sizes,
        int const* __restrict__ comm_ext,
        int const* __restrict__ comm_int,
        int *__restrict__ sources,
        int2 *__restrict__ insert_improvements,
        int *__restrict__ remove_improvements) {
    int i = get_global_id();
    if (i >= num_labels) return;

    int vertex = vertex_label_pairs[i].x;
    int new_label = vertex_label_pairs[i].y;
    int freq = freqs[i];
    int degree = indices[vertex + 1] - indices[vertex];

    int old_label = labels[vertex];

    int size = comm_sizes[new_label];
    int ext_edges = comm_ext[new_label];
    int int_edges = comm_int[new_label] / 2;

    if (old_label == new_label) {
        size--;
        ext_edges += (freq - (degree - freq));
        int_edges -= freq;
    }

    if (new_label == INVALID_LABEL) {
        size = 1;
        ext_edges = 0;
        int_edges = 0;
    }

    double p_int = 0.0;

    if (size > 1) {
        p_int = (2.0 * int_edges) / (size * (size - 1.0));
    }


    double imp = calculate_improvement(
        size,
        freq,
        degree - freq,
        ext_edges,
        p_int,
        clustering_coef,
        1.0);

    if (vertex == 24 && (new_label == 99 || new_label == 101571)) {
        printf("50] %d %d %f %d\n", new_label, freq, imp, size);
    }

    sources[i] = vertex;
    insert_improvements[i] = make_int2(new_label, pack_float((float) imp));

    if (old_label == new_label) {
        remove_improvements[vertex] = pack_float((float) -imp);
    }
}


__global__ void update_labels(
        const int n,
        int2 const *__restrict__ vertex_insert_improv,
        int const *__restrict__ vertex_remove_improv,
        int *__restrict__ labels) {
    int i = get_global_id();
    if (i >= n) return;

    float remove_improv = unpack_float(vertex_remove_improv[i]);
    float insert_improv = unpack_float(vertex_insert_improv[i].y)
        + remove_improv;
    int insert_label = vertex_insert_improv[i].x;

    int new_label;

    if (insert_improv > 0 && insert_improv > remove_improv) {
        new_label = insert_label;
    } else if (remove_improv > 0) {
        new_label = INVALID_LABEL;
    } else {
        return;
    }

    labels[i] = new_label;
}

__global__ void update_comm_sizes(
        const int n,
        int const* __restrict__ labels,
        int *__restrict__ comm_sizes) {
    int i = get_global_id();
    if (i >= n) return;

    int l = labels[i];
    if (l == INVALID_LABEL) return;

    atomicAdd(&comm_sizes[l], 1);
}


__global__ void update_comm_edges(
        const int m,
        int const* __restrict__ labels,
        int const* __restrict__ sources,
        int const* __restrict__ adj,
        int *__restrict__ comm_int,
        int *__restrict__ comm_ext) {
    int i = get_global_id();
    if (i >= m) return;

    int a = labels[sources[i]];
    int b = labels[adj[i]];

    if (a == INVALID_LABEL) return;

    if (a == b) {
        atomicAdd(&comm_int[a], 1);
    } else {
        atomicAdd(&comm_ext[a], 1);
    }
}


__global__ void compute_wccs(
        const int n,
        int const* __restrict__ tri,
        int const* __restrict__ tri_degs,
        int const* __restrict__ int_tri,
        int const* __restrict__ int_tri_degs,
        int const* __restrict__ labels,
        int const* __restrict__ comm_sizes,
        double *wcc) {
    int i = get_global_id();
    if (i >= n) return;

    int t = tri[i];
    int td = tri_degs[i];
    int it = int_tri[i];
    int itd = int_tri_degs[i];
    int l = labels[i];
    int size = l != INVALID_LABEL ? comm_sizes[l] : 1;

    wcc[i] = t == 0 ? 0.0 :
        (it / ((double) t)) * (td / ((double) (td + (size - 1) - itd)));
}




__global__ void set_internal_edges(
        const int m,
        int const* __restrict__ labels,
        int const *__restrict__ sources,
        int const *__restrict__ adj,
        int *__restrict__ adj_indices) {
    int i = get_global_id();
    if (i >= m) return;

    int la = labels[sources[i]];
    int lb = labels[adj[i]];

    adj_indices[i] = (la != INVALID_LABEL && la == lb);

}

__global__ void create_indices(
        const int n,
        int const *__restrict__ adj_indices,
        int const *__restrict__ old_indices,
        int *__restrict__ new_indices) {
    int i = get_global_id();
    if (i >= n + 1) return;

    new_indices[i] = adj_indices[old_indices[i]];
}

__global__ void create_adj(
        const int m,
        int const *__restrict__ adj_indices,
        int const *__restrict__ old_sources,
        int const *__restrict__ old_adj,
        int *__restrict__ new_sources,
        int *__restrict__ new_adj) {
    int i = get_global_id();
    if (i >= m) return;

    int ia = adj_indices[i];
    int ib = adj_indices[i + 1];

    if (ia != ib) {
        new_adj[ia] = old_adj[i];
        new_sources[ia] = old_sources[i];
    }
}

__global__ void count_triangles(
        const int m,
        const int n,
        int const* __restrict__ labels,
        int const* __restrict__ sources,
        int const* __restrict__ adj,
        int const* __restrict__ indices,
        int *__restrict__ int_tri,
        int *__restrict__ int_tri_deg) {
    int id = get_global_id();
    if (id >= m) return;

    int a = sources[id];
    int b = adj[id];
    int l = labels[a];


    if (a > b || a >= n || l == INVALID_LABEL || l != labels[b]) {
        return;
    }

    int i = indices[a];
    int j = indices[b];

    int i_end = indices[a + 1];
    int j_end = indices[b + 1];

    int p = adj[i];
    int q = adj[j];

    int triangles = 0;

    while (i < i_end && j < j_end) {
        int d = p - q;

        if (d == 0) {
            triangles++;
        }

        if (d <= 0) {
            i++;
            p = i < i_end ? adj[i] : 0;
        }

        if (d >= 0) {
            j++;
            q = j < j_end ? adj[j] : 0;
        }
    }

    if (triangles > 0) {
        if (a < n) atomicAdd(&int_tri[a], triangles);
        if (b < n) atomicAdd(&int_tri[b], triangles);

        if (a < n) atomicAdd(&int_tri_deg[a], 1);
        if (b < n) atomicAdd(&int_tri_deg[b], 1);
    }
}

__global__ void prepare_outgoing_ghosts(
        int num,
        int const* __restrict__ labels,
        int const* __restrict__ ghost_vertices,
        int *__restrict__ ghost_labels) {
    int id = get_global_id();
    if (id >= num) return;

    ghost_labels[id] = labels[ghost_vertices[id]];
}
