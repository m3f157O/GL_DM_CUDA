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

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

//////////////////////////////
//////////////////////////////

// Pointers for converted vector




// Write GPU kernel here!

//////////////////////////////
//////////////////////////////

__global__ void spmv_coo_gpu(const int *x_gpu, const int *y_gpu, const double *val_gpu, const double *pr_gpu, double *pr_tmp_gpu, int E){


    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i=thread_id;i<E;i+=blockDim.x*gridDim.x){
        atomicAdd(&pr_tmp_gpu[x_gpu[i]], val_gpu[i] * pr_gpu[y_gpu[i]]);

    }


}

__global__ void dot_product_gpu(int *dangling_gpu, double *pr_gpu, int *V, double *dangling_factor_gpu) {

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i=thread_id;i<(*V);i+=blockDim.x*gridDim.x){
        atomicAdd(dangling_factor_gpu, dangling_gpu[i] * pr_gpu[i]);

    }


}

__global__ void calculateBeta(double *beta, double *dangling, double alpha, int *V){

    (*beta) = (*dangling) * alpha / (*V);
}

__global__ void initKernel(double *pr_gpu, int len){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<len){
        pr_gpu[i] = 1.0/len;

    }



}

__global__ void axbp_custom(double alpha, double one_minus_a, double* pr_tmp, double *beta, int personalization_vertex, double* pr_tmp_result, int *V)
{



    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i=thread_id;i<(*V);i+=blockDim.x*gridDim.x){
        pr_tmp_result[i]=alpha * pr_tmp[i] + (*beta) + ((personalization_vertex == i) ? one_minus_a : 0.0);

    }

}

__global__ void euclidean_distance_gpu(double *err,double *pr, double *pr_tmp, int *V) {


    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i=thread_id;i<(*V);i+=blockDim.x*gridDim.x){
        double temp = pr[i]-pr_tmp[i];
        atomicAdd(err, temp*temp);
    }


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
    dangling_size = V * sizeof(int); //ok
    E_size = E * sizeof(int);
    val_size = E * sizeof(double);



    // Allocate space in VRAM
    cudaMalloc((void **)&x_gpu, E_size);
    cudaMalloc((void **)&y_gpu, E_size);
    cudaMalloc((void **)&val_gpu, val_size);
    cudaMalloc((void **)&pr_gpu, V_size);
    cudaMalloc((void **)&pr_tmp_gpu, V_size);
    cudaMalloc((void **)&dangling_gpu, dangling_size);
    cudaMalloc((void **)&dangling_factor_gpu, sizeof(double));
    cudaMalloc((void **)&err_gpu, sizeof(double));
    cudaMalloc((void **)&beta_gpu, sizeof(double));
    cudaMalloc((void **)&V_gpu, sizeof(int));
}

// Initialize data;
void PersonalizedPageRank::init() {
    // Do any additional CPU or GPU setup here;

}

// Reset the state of the computation after every iteration.
// Reset the result, and transfer data to the GPU if necessary;
void PersonalizedPageRank::reset() {
    // Reset the PageRank vector (uniform initialization, 1 / V for each vertex);
    //std::fill(pr.begin(), pr.end(), 1.0 / V);
    // Generate a new personalization vertex for this iteration;
    personalization_vertex = rand() % V;
    if (debug) std::cout << "personalization vertex=" << personalization_vertex << std::endl;
    //std::cout << "RESET";

    x_array = &x[0];
    y_array = &y[0];
    dangling_array = &dangling[0];
    pr_array = &pr[0];
    val_array = &val[0];
    //std::vector<double> pr_tmp;
    //pr_tmp_array = &pr_tmp[0];
    /*for(int i=0;i<V;i++{}
        std::cout << val_array[i] ;
        std::cout << "THIS IS THE DOG\n";

    }*/





    // Do any GPU reset here, and also transfer data to the GPU;
    cudaMemcpy(pr_gpu, pr_array, V_size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_gpu, y_array, E_size, cudaMemcpyHostToDevice);
    cudaMemcpy(val_gpu, val_array, val_size, cudaMemcpyHostToDevice);
    cudaMemcpy(x_gpu, x_array, E_size, cudaMemcpyHostToDevice);
    cudaMemcpy(V_gpu, &V, sizeof(int), cudaMemcpyHostToDevice);
    /*cudaError_t err = */cudaMemcpy(dangling_gpu, dangling_array, dangling_size, cudaMemcpyHostToDevice);



    //cudaError_t err = cudaGetLastError();

    /*if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to copy IN RESET from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }*/
    // Copy data from RAM to VRAM



}

void PersonalizedPageRank::execute(int iteration) {
    // Do the GPU computation here, and also transfer results to the CPU;

    max_iterations = 100;
    convergence_threshold = 1e-6;

    //make block size for E
    int threads_per_block = 512;
    int num_blocks = (E + threads_per_block - 1)/ threads_per_block;

    //make block size for V

    int threads_per_block_vertex = 64;
    int num_blocks_vertex = (V + threads_per_block_vertex - 1)/ threads_per_block_vertex;

    double *temp;
    int iter = 0;
    bool converged = false;
    while (!converged && iter < max_iterations) {

        auto start_tmp = clock_type::now();
        auto end_tmp = clock_type::now();

        start_tmp = clock_type::now();
        cudaMemset(pr_tmp_gpu, 0, V_size);
        cudaMemset(dangling_factor_gpu, 0, sizeof(double));
        cudaMemset(err_gpu, 0, sizeof(double));
        end_tmp = clock_type::now();
        if(iter==0) std::cout << "memset time: " << float(chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count()) / 1000 << "ms\n";

        start_tmp = clock_type::now();
        spmv_coo_gpu<<<num_blocks,threads_per_block>>>(x_gpu, y_gpu, val_gpu, pr_gpu, pr_tmp_gpu, E);
        end_tmp = clock_type::now();
        if(iter==0) std::cout << "spmv_coo time: " << float(chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count()) / 1000 << "ms\n";

        start_tmp = clock_type::now();
        dot_product_gpu<<<num_blocks_vertex,threads_per_block_vertex>>>(dangling_gpu, pr_gpu, V_gpu, dangling_factor_gpu);
        end_tmp = clock_type::now();
        if(iter==0) std::cout << "dot_product time: " << float(chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count()) / 1000 << "ms\n";

        calculateBeta<<<1,1>>>(beta_gpu, dangling_factor_gpu, alpha, V_gpu);

        start_tmp = clock_type::now();
        axbp_custom<<<num_blocks_vertex,threads_per_block_vertex>>>(alpha, 1-alpha, pr_tmp_gpu, beta_gpu, personalization_vertex, pr_tmp_gpu, V_gpu);
        end_tmp = clock_type::now();
        if(iter==0) std::cout << "axbp time: " << float(chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count()) / 1000 << "ms\n";

        start_tmp = clock_type::now();
        euclidean_distance_gpu<<<num_blocks_vertex,threads_per_block_vertex>>>(err_gpu, pr_gpu, pr_tmp_gpu, V_gpu);
        end_tmp = clock_type::now();
        if(iter==0) std::cout << "euclidean_distance time: " << float(chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count()) / 1000 << "ms\n";

        start_tmp = clock_type::now();
        if(iter>max_iterations/10 && iter%5==0){
            cudaMemcpy(&error, err_gpu, sizeof(double), cudaMemcpyDeviceToHost);
            converged = std::sqrt(error) <= convergence_threshold;
        }
        end_tmp = clock_type::now();
        if(iter==15) std::cout << "error management time: " << float(chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count()) / 1000 << "ms\n";

        // Update the PageRank vector;

        temp=pr_gpu;
        pr_gpu=pr_tmp_gpu;
        pr_tmp_gpu=temp;


        //cudaMemcpy(pr_gpu, pr_tmp_gpu, V_gpu_size ,cudaMemcpyDeviceToDevice);
        iter++;
    }
    cudaMemcpy(pr_array, pr_gpu, V_size, cudaMemcpyDeviceToHost);

}

void PersonalizedPageRank::cpu_validation(int iter) {

    // Reset the CPU PageRank vector (uniform initialization, 1 / V for each vertex);

    ////// INITIALIZE DUMMY VECTOR WITH 1/V FLOAT
    std::fill(pr_golden.begin(), pr_golden.end(), 1.0 / V);

    // Do Personalized PageRank on CPU;
    auto start_tmp = clock_type::now();
    /////// pass x (starting edge vertex) y (arriving edge vertex)  OK
    /// val ( is all one at starting )
    /// pr_golden (pagerank scores)
    personalized_pagerank_cpu(x.data(), y.data(), val.data(), V, E, pr_golden.data(), dangling.data(), personalization_vertex, alpha, 1e-6, 100);
    auto end_tmp = clock_type::now();
    auto exec_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
    std::cout << "exec time CPU=" << double(exec_time) / 1000 << " ms" << std::endl;

    // Obtain the vertices with highest PPR value;
    std::vector<std::pair<int, double>> sorted_pr_tuples = sort_pr(pr.data(), V);
    std::vector<std::pair<int, double>> sorted_pr_golden_tuples = sort_pr(pr_golden.data(), V);

    // Check how many of the correct top-20 PPR vertices are retrieved by the GPU;
    std::unordered_set<int> top_pr_indices;
    std::unordered_set<int> top_pr_golden_indices;
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
            } else if (std::abs(sorted_pr_tuples[i].second - sorted_pr_golden_tuples[i].second) > 1e-6) {
                std::cout << "* error in value! (" << i << ") correct=" << pr_id_cpu << " (val=" << pr_val_cpu << "), found=" << pr_id_gpu << " (val=" << pr_val_gpu << ")" << std::endl;
            }
        }
    }
    std::cout.precision(old_precision);
    // Set intersection to find correctly retrieved vertices;
    std::vector<int> correctly_retrieved_vertices;
    set_intersection(top_pr_indices.begin(), top_pr_indices.end(), top_pr_golden_indices.begin(), top_pr_golden_indices.end(), std::back_inserter(correctly_retrieved_vertices));
    precision = double(correctly_retrieved_vertices.size()) / topk;
    if (debug) std::cout << "correctly retrived top-" << topk << " vertices=" << correctly_retrieved_vertices.size() << " (" << 100 * precision << "%)" << std::endl;
}

std::string PersonalizedPageRank::print_result(bool short_form) {
    if (short_form) {
        return std::to_string(precision);
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
    cudaFree(dangling_gpu);
    cudaFree(dangling_factor_gpu);
    cudaFree(err_gpu);
    cudaFree(beta_gpu);
    cudaFree(V_gpu);
}
