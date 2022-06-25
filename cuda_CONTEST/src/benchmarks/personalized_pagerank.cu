// Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NECSTLab nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//  * Neither the name of Politecnico di Milano nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <sstream>
#include "personalized_pagerank.cuh"
#include "cublas_v2.h"

#define DEFAULT_THREADS_PER_BLOCK_VERTEX 64
#define DEFAULT_THREADS_PER_BLOCK 1024

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

//////////////////////////////
//////////////////////////////

// Pointers for converted vector




// Write GPU kernel here!

//////////////////////////////
//////////////////////////////

__global__ void spmv_coo_gpu(const int *x_gpu, const int *y_gpu, const double *val_gpu, const double *pr_gpu, double *pr_tmp_gpu, int *E, int *V) {

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i=thread_id;i<(*V);i+=blockDim.x*gridDim.x){
        pr_tmp_gpu[i] = 0;
    }
    __syncthreads();
    for(int i=thread_id;i<(*E);i+=blockDim.x*gridDim.x) {
        atomicAdd(&pr_tmp_gpu[x_gpu[i]], val_gpu[i] * pr_gpu[y_gpu[i]]);
    }
}

__global__ void spmv_coo_gpu_2(const int *x_gpu, const int *y_gpu, const double *val_gpu, const double *pr_gpu, double *pr_tmp_gpu, int *E, int *V) {

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i=thread_id;i<(*V);i+=blockDim.x*gridDim.x){
        pr_tmp_gpu[i] = 0;
    }
    __syncthreads();
    int tid = threadIdx.x;
    int i = tid+blockIdx.x*blockDim.x;
    if(i<(*E)) {
        __shared__ double temp[DEFAULT_THREADS_PER_BLOCK];
        __shared__ int idx[DEFAULT_THREADS_PER_BLOCK];
        temp[tid] = val_gpu[i] * pr_gpu[y_gpu[i]];
        idx[tid] = x_gpu[i];
        __syncthreads();
        atomicAdd(&pr_tmp_gpu[idx[tid]], temp[tid]);
    }
}

__global__ void dot_product_gpu(int *dangling_gpu, double *pr_gpu, int *V, double *dangling_factor_gpu) {

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if(thread_id == 0) (*dangling_factor_gpu)=0;
    for(int i=thread_id;i<(*V);i+=blockDim.x*gridDim.x){
        atomicAdd(dangling_factor_gpu, dangling_gpu[i] * pr_gpu[i]);
    }
}

__global__ void dot_product_gpu_2(int *dangling, double *pr, int *V, double *dangling_factor) {

    int tid = threadIdx.x;
    int i = tid+blockIdx.x*blockDim.x;

    if(i == 0) (*dangling_factor)=0;

    if(i<(*V)) {
        __shared__ double temp[DEFAULT_THREADS_PER_BLOCK_VERTEX];
        temp[tid] = dangling[i] * pr[i];
        __syncthreads();
        for(unsigned int s = 1; s < blockDim.x; s *= 2) {
            int index = 2 * s * tid;
            if (index < blockDim.x) {
                temp[index] += temp[index + s];
            }
        }

        if(tid == 0) atomicAdd(dangling_factor, temp[0]);
    }
}

__global__ void calculateBeta(double *beta, double *dangling_factor, double *alpha, int *V) {

    (*beta) = (*dangling_factor) * (*alpha) / (*V);
}

__global__ void axbp_custom(double *alpha, double *pr_tmp, double *beta, int *personalization_vertex, int *V) {

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i=thread_id;i<(*V);i+=blockDim.x*gridDim.x){
        pr_tmp[i]=(*alpha) * pr_tmp[i] + (*beta) + ((*personalization_vertex == i) ? (1-(*alpha)) : 0.0);
    }

}

__global__ void euclidean_distance_gpu(double *err,double *pr, double *pr_tmp, int *V) {

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if(thread_id == 0) (*err)=0;
    for(int i=thread_id;i<(*V);i+=blockDim.x*gridDim.x){
        atomicAdd(err, (pr[i]-pr_tmp[i])*(pr[i]-pr_tmp[i]));
    }

}

__global__ void euclidean_distance_gpu_2(double *err, double *pr, double *pr_tmp, int *V) {

    int tid = threadIdx.x;
    int i = tid+blockIdx.x*blockDim.x;

    if(i == 0) (*err)=0;
    if(i<(*V)) {
        __shared__ double temp[DEFAULT_THREADS_PER_BLOCK_VERTEX];
        temp[tid] = (pr[i] - pr_tmp[i]) * (pr[i] - pr_tmp[i]);
        __syncthreads();
        for(unsigned int s = 1; s < blockDim.x; s *= 2) {
            int index = 2 * s * tid;
            if (index < blockDim.x) {
                temp[index] += temp[index + s];
            }
        }

        if(tid == 0) atomicAdd(err, temp[0]);
    }
}

__global__ void axbp_euclidean_distance_gpu(double *alpha, double *pr_tmp, double *beta, int *personalization_vertex, double *err, double *pr, int *V, bool *converged, double convergence_threshold) {
    int tid = threadIdx.x;
    int i = tid+blockIdx.x*blockDim.x;

    if(i == 0) {
        (*err)=0;
        (*converged)=false;
    }
    if(i<(*V)) {
        pr_tmp[i]=(*alpha) * pr_tmp[i] + (*beta) + ((*personalization_vertex == i) ? (1-(*alpha)) : 0.0);
        __shared__ double temp[DEFAULT_THREADS_PER_BLOCK_VERTEX];
        temp[tid] = (pr[i] - pr_tmp[i]) * (pr[i] - pr_tmp[i]);
        __syncthreads();
        for(unsigned int s = 1; s < blockDim.x; s *= 2) {
            int index = 2 * s * tid;
            if (index < blockDim.x) {
                temp[index] += temp[index + s];
            }
        }

        if(tid == 0) atomicAdd(err, temp[0]);
        __syncthreads();
    }
    if(i == 0)
        (*converged) = sqrt(*err) <= convergence_threshold;

}

__device__ double dangling_factor;
__device__ double beta;
__device__ double error;
__device__ bool converged;
__global__ void main_kernel(int *x, int *y, double *val, double *pr, double *pr_tmp, int *dangling, double *alpha, int *personalization_vertex, int max_iterations, double convergence_threshold, int *E, int *V) {

    cudaStream_t s1, s2;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);

    //make block size for E
    int threads_per_block = DEFAULT_THREADS_PER_BLOCK;
    int num_blocks = ((*E) + threads_per_block - 1)/ threads_per_block;

    //make block size for V
    int threads_per_block_vertex = DEFAULT_THREADS_PER_BLOCK_VERTEX;
    int num_blocks_vertex = ((*V) + threads_per_block_vertex - 1)/ threads_per_block_vertex;

    int iter = 0;
    converged = false;
    double *temp;

    while (!converged && iter<max_iterations) {

        spmv_coo_gpu<<<num_blocks,threads_per_block,0,s1>>>(x, y, val, pr, pr_tmp, E, V);

        dot_product_gpu_2<<<num_blocks_vertex,threads_per_block_vertex,0,s2>>>(dangling, pr, V, &dangling_factor);

        cudaDeviceSynchronize();
        beta = dangling_factor * (*alpha) / (*V);

        axbp_euclidean_distance_gpu<<<num_blocks_vertex,threads_per_block_vertex>>>(alpha, pr_tmp, &beta, personalization_vertex, &error, pr, V, &converged, convergence_threshold);

        cudaDeviceSynchronize();
        // Update the PageRank vector;
        temp=pr;
        pr=pr_tmp;
        pr_tmp=temp;

        iter++;
    }
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
}

// CPU Utility functions;

// Read the input graph and initialize it;
void PersonalizedPageRank::initialize_graph() {
    // Read the graph from an MTX file;
    int num_rows = 0;
    int num_columns = 0;
    read_mtx(graph_file_path.c_str(), &x, &y, &val,
        &num_rows, &num_columns, &E, // Store the number of vertices (row and columns must be the same value), and edges;
        true,                        // If true, read edges TRANSPOSED, i.e. edge (2, 3) is loaded as (3, 2). We set this true as it simplifies the PPR computation;
        false,                       // If true, read the third column of the matrix file. If false, set all values to 1 (this is what you want when reading a graph topology);
        debug,                 
        false,                       // MTX files use indices starting from 1. If for whatever reason your MTX files uses indices that start from 0, set zero_indexed_file=true;
        true                         // If true, sort the edges in (x, y) order. If you have a sorted MTX file, turn this to false to make loading faster;
    );
    if (num_rows != num_columns) {
        if (debug) std::cout << "error, the matrix is not squared, rows=" << num_rows << ", columns=" << num_columns << std::endl;
        exit(-1);
    } else {
        V = num_rows;
    }
    if (debug) std::cout << "loaded graph, |V|=" << V << ", |E|=" << E << std::endl;

    // Compute the dangling vector. A vertex is not dangling if it has at least 1 outgoing edge;
    dangling.resize(V);
    std::fill(dangling.begin(), dangling.end(), 1);  // Initially assume all vertices to be dangling;
    for (int i = 0; i < E; i++) {
        // Ignore self-loops, a vertex is still dangling if it has only self-loops;
        if (x[i] != y[i]) dangling[y[i]] = 0;
    }
    // Initialize the CPU PageRank vector;
    pr.resize(V);
    pr_golden.resize(V);
    // Initialize the value vector of the graph (1 / outdegree of each vertex).
    // Count how many edges start in each vertex (here, the source vertex is y as the matrix is transposed);
    int *outdegree = (int *) calloc(V, sizeof(int));
    for (int i = 0; i < E; i++) {
        outdegree[y[i]]++;   /////this is done in initKernelOutdegree
        //printf("%d",outdegree[y[i]]);
    }
    // Divide each edge value by the outdegree of the source vertex;
    for (int i = 0; i < E; i++) {
    //// each node val is dependent on the number of outgoing edges
        val[i] = 1.0 / outdegree[y[i]];  /////this is done in initKernelInverse
    }
    free(outdegree);
}

//////////////////////////////
//////////////////////////////

// Allocate data on the CPU and GPU;
void PersonalizedPageRank::alloc() {
    // Load the input graph and preprocess it;
    initialize_graph();

    // Size of allocations
    V_size = V * sizeof(double);
    dangling_size = V * sizeof(int);
    E_size = E * sizeof(int);
    val_size = E * sizeof(double);



    // Allocate space in VRAM
    cudaMalloc((void **)&x_gpu, E_size);
    cudaMalloc((void **)&y_gpu, E_size);
    cudaMalloc((void **)&val_gpu, val_size);
    cudaMalloc((void **)&pr_gpu, V_size);
    cudaMalloc((void **)&pr_tmp_gpu, V_size);
    cudaMalloc((void **)&alpha_gpu, sizeof(double));
    cudaMalloc((void **)&V_gpu, sizeof(int));
    cudaMalloc((void **)&E_gpu, sizeof(int));
    cudaMalloc((void **)&personalization_vertex_gpu, sizeof(int));

    if(implementation != 1) {
        cudaMalloc((void **)&dangling_gpu, dangling_size);
    } else {
        cudaMalloc((void **)&dangling_factor_gpu, sizeof(double));
        cudaMalloc((void **)&beta_gpu, sizeof(double));
        cudaMalloc((void **)&error_gpu, sizeof(double));
        cudaMalloc((void **)&temp_gpu, sizeof(double));
        cudaMalloc((void **)&dangling_double_gpu, V_size);
        cudaMalloc((void **)&converged_gpu, sizeof(bool));
    }

}

// Initialize data;
void PersonalizedPageRank::init() {
    // Do any additional CPU or GPU setup here;

    cudaMemcpy(x_gpu, &x[0], E_size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_gpu, &y[0], E_size, cudaMemcpyHostToDevice);
    cudaMemcpy(val_gpu, &val[0], val_size, cudaMemcpyHostToDevice);
    cudaMemcpy(alpha_gpu, &alpha, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(V_gpu, &V, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(E_gpu, &E, sizeof(int), cudaMemcpyHostToDevice);

    if(implementation != 1) {
        cudaMemcpy(dangling_gpu, &dangling[0], dangling_size, cudaMemcpyHostToDevice);
    } else {
        std::vector<double> dangling_tmp(dangling.begin(), dangling.end());
        cudaMemcpy(dangling_double_gpu, &dangling_tmp[0], V * sizeof(double), cudaMemcpyHostToDevice);
    }
}

// Reset the state of the computation after every iteration.
// Reset the result, and transfer data to the GPU if necessary;
void PersonalizedPageRank::reset() {
    // Reset the PageRank vector (uniform initialization, 1 / V for each vertex);
    std::fill(pr.begin(), pr.end(), 1.0 / V);
    cudaMemcpy(pr_gpu, &pr[0], V_size, cudaMemcpyHostToDevice);

    // Generate a new personalization vertex for this iteration;
    personalization_vertex = rand() % V;
    if (debug) std::cout << "personalization vertex=" << personalization_vertex << std::endl;

    // Do any GPU reset here, and also transfer data to the GPU;
    cudaMemcpy(personalization_vertex_gpu, &personalization_vertex, sizeof(int), cudaMemcpyHostToDevice);
}

void PersonalizedPageRank::execute(int iteration) {
    // Do the GPU computation here, and also transfer results to the CPU;

    if(implementation != 1) {
        main_kernel<<<1,1>>>(x_gpu, y_gpu, val_gpu, pr_gpu, pr_tmp_gpu, dangling_gpu, alpha_gpu, personalization_vertex_gpu, max_iterations, convergence_threshold, E_gpu, V_gpu);

    } else {
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

        //make block size for E
        int threads_per_block = DEFAULT_THREADS_PER_BLOCK;
        int num_blocks = (E + threads_per_block - 1)/ threads_per_block;

        //make block size for V
        int threads_per_block_vertex = DEFAULT_THREADS_PER_BLOCK_VERTEX;
        int num_blocks_vertex = (V + threads_per_block_vertex - 1)/ threads_per_block_vertex;

        int iter = 0;
        bool converged = false;

        while (!converged && iter<max_iterations) {

            spmv_coo_gpu<<<num_blocks,threads_per_block>>>(x_gpu, y_gpu, val_gpu, pr_gpu, pr_tmp_gpu, E_gpu, V_gpu);

            cublasDdot(handle, V, dangling_double_gpu, 1, pr_gpu, 1, dangling_factor_gpu);

            calculateBeta<<<1,1>>>(beta_gpu, dangling_factor_gpu, alpha_gpu, V_gpu);

            axbp_euclidean_distance_gpu<<<num_blocks_vertex,threads_per_block_vertex>>>(alpha_gpu, pr_tmp_gpu, beta_gpu, personalization_vertex_gpu, error_gpu, pr_gpu, V_gpu, converged_gpu, convergence_threshold);

            cudaMemcpy(&converged, converged_gpu, sizeof(bool), cudaMemcpyDeviceToHost);

            // Update the PageRank vector;
            temp_gpu=pr_gpu;
            pr_gpu=pr_tmp_gpu;
            pr_tmp_gpu=temp_gpu;

            iter++;
        }
        cublasDestroy(handle);
    }

    cudaMemcpy(&pr[0], pr_gpu, V_size, cudaMemcpyDeviceToHost);
}

void PersonalizedPageRank::cpu_validation(int iter) {

    // Reset the CPU PageRank vector (uniform initialization, 1 / V for each vertex);
    std::fill(pr_golden.begin(), pr_golden.end(), 1.0 / V);

    // Do Personalized PageRank on CPU;
    auto start_tmp = clock_type::now();
    personalized_pagerank_cpu(x.data(), y.data(), val.data(), V, E, pr_golden.data(), dangling.data(), personalization_vertex, alpha, convergence_threshold, max_iterations);
    auto end_tmp = clock_type::now();
    auto exec_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
    if(debug) std::cout << "exec time CPU=" << double(exec_time) / 1000 << " ms" << std::endl;

    // Obtain the vertices with highest PPR value;
    std::vector<std::pair<int, double>> sorted_pr_tuples = sort_pr(pr.data(), V);
    std::vector<std::pair<int, double>> sorted_pr_golden_tuples = sort_pr(pr_golden.data(), V);

    // Check how many of the correct top-20 PPR vertices are retrieved by the GPU;
    std::set<int> top_pr_indices;
    std::set<int> top_pr_golden_indices;
    int old_precision = std::cout.precision();
    std::cout.precision(4);
    int topk = std::min(V, topk_vertices);
    for (int i = 0; i < topk; i++) {
        int pr_id_gpu = sorted_pr_tuples[i].first;
        int pr_id_cpu = sorted_pr_golden_tuples[i].first;
        top_pr_indices.insert(pr_id_gpu);
        top_pr_golden_indices.insert(pr_id_cpu);
        if (debug) {
            double pr_val_gpu = sorted_pr_tuples[i].second;
            double pr_val_cpu = sorted_pr_golden_tuples[i].second;
            if (pr_id_gpu != pr_id_cpu) {
                std::cout << "* error in rank! (" << i << ") correct=" << pr_id_cpu << " (val=" << pr_val_cpu << "), found=" << pr_id_gpu << " (val=" << pr_val_gpu << ")" << std::endl;
            } else if (std::abs(sorted_pr_tuples[i].second - sorted_pr_golden_tuples[i].second) > convergence_threshold) {
                std::cout << "* error in value! (" << i << ") correct=" << pr_id_cpu << " (val=" << pr_val_cpu << "), found=" << pr_id_gpu << " (val=" << pr_val_gpu << ")" << std::endl;
            }
        }
    }
    precision_to_print = precision;
    std::cout.precision(old_precision);
    // Set intersection to find correctly retrieved vertices;
    std::vector<int> correctly_retrieved_vertices;
    set_intersection(top_pr_indices.begin(), top_pr_indices.end(), top_pr_golden_indices.begin(), top_pr_golden_indices.end(), std::back_inserter(correctly_retrieved_vertices));
    precision = double(correctly_retrieved_vertices.size()) / topk;
    if (debug) std::cout << "correctly retrived top-" << topk << " vertices=" << correctly_retrieved_vertices.size() << " (" << 100 * precision << "%)" << std::endl;
}

std::string PersonalizedPageRank::print_result(bool short_form) {
    if (short_form) {
        return std::to_string(precision_to_print);
    } else {
        // Print the first few PageRank values (not sorted);
        std::ostringstream out;
        out.precision(3);
        out << "[";
        for (int i = 0; i < std::min(20, V); i++) {
            out << pr[i] << ", ";
        }
        out << "...]";
        return out.str();
    }
}


void PersonalizedPageRank::clean() {
    // Delete any GPU data or additional CPU data;
    cudaFree(x_gpu);
    cudaFree(y_gpu);
    cudaFree(val_gpu);
    cudaFree(pr_gpu);
    cudaFree(pr_tmp_gpu);
    cudaFree(V_gpu);
    cudaFree(alpha_gpu);
    cudaFree(personalization_vertex_gpu);
    cudaFree(E_gpu);

    if(implementation != 1) {
        cudaFree(dangling_gpu);
    } else {
        cudaFree(dangling_factor_gpu);
        cudaFree(beta_gpu);
        cudaFree(error_gpu);
        cudaFree(temp_gpu);
        cudaFree(dangling_double_gpu);
        cudaFree(converged_gpu);
    }
}
