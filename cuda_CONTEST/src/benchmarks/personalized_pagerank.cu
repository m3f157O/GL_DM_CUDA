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

int* x_array;
int* y_array;
double* pr_array;
// Pointers for VRAM data
int *x_gpu, *y_gpu;
double *val_gpu, *pr_gpu;

double *pr_tmp_gpu;
int V_size;
int E_size;

// Write GPU kernel here!

//////////////////////////////
//////////////////////////////
__global__ void cuda_hello(){
    printf("this is working");

}
/*
 * inline void spmv_coo_cpu(const int *x, const int *y, const double *val, const double *vec, double *result, int N) {
    for (int i = 0; i < N; i++) {
        //// the value of each node is the summation of the value of outgoing nodes * the previous pagerank score
        result[x[i]] += val[i] * vec[y[i]];
    }
}
 */
__global__ void spmv_coo_gpu(const int *x_gpu, const int *y_gpu, const double *val_gpu, const double *pr_gpu, double *pr_tmp_gpu, int V){


    //val_gpu, , pr_gpu, pr_tmp_gpu are all OK
  //  printf("THIS IS VAL\n");
    int i = threadIdx.x + blockIdx.x * blockDim.x;
 //   printf("%f\n",val_gpu[i]);
    printf("%d\n",i);
    //printf("%f\n",atomicAdd( & pr_tmp_gpu[x_gpu[i]], 1 ));

    printf("THIS IS X GPU "); //read correctly
    printf("%d\n",x_gpu[i]);

    printf("THIS IS Y GPU "); //read correctly
    printf("%d\n",y_gpu[i]);

    printf("THIS IS PR GPU ");
    printf("%f\n",pr_gpu[i]);

    printf("THIS IS PR TEMP GPU ");
    printf("%f\n",pr_tmp_gpu[i]);


    printf("THIS IS VAL GPU ");
    printf("%f\n",pr_tmp_gpu[i]);

    pr_tmp_gpu[x_gpu[i]] += val_gpu[i] * pr_gpu[y_gpu[i]];  //explodes
    //pr_tmp_gpu[x_gpu[i]] += val_gpu[i] * pr_gpu[y_gpu[i]];
   /* printf("%d\n",i);
    //printf("%f\n",atomicAdd( & pr_tmp_gpu[x_gpu[i]], 1 ));

    printf("THIS IS X GPU\n");
    printf("%d\n",x_gpu[i]);

    printf("THIS IS PR GPU\n");
    printf("%f\n",pr_gpu[i]);

    printf("THIS IS PR TEMP GPU\n");
    printf("%f\n",pr_tmp_gpu[i]);*/
}


// CPU Utility functions;
void personalized_pagerank_gpu_support(
        const int V,
        const int E,
        const int *dangling_bitmap,
        const int personalization_vertex,
        double *pr,
        double *val,
        double alpha=DEFAULT_ALPHA,
        double convergence_threshold=DEFAULT_CONVERGENCE,
        const int max_iterations=DEFAULT_MAX_ITER
        ){

    // Temporary PPR result;

    int iter = 0;  //stay on cpu
    bool converged = false; //stay on cpu
    while (!converged && iter < max_iterations) {     //stay on cpu


        cudaMemset(pr_tmp_gpu, 0, V_size);  // ???

        int N = V;
        int threads_per_block = std::min(32,V);   //todo make sure that num blocks is always >0
        int num_blocks = N / threads_per_block;
        // Launch add() kernel on GPU

        //spmv_coo_gpu<<<num_blocks,threads_per_block>>>(x_gpu, y_gpu, val_gpu, pr_gpu, pr_tmp_gpu,V);
        spmv_coo_gpu<<<num_blocks,threads_per_block>>>(x_gpu, y_gpu, val_gpu, pr_gpu, pr_tmp_gpu,V);


        cudaDeviceSynchronize();



        cudaError_t err = cudaGetLastError();
       /* if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));*/

        /*////get dangling vector percentage
        double dangling_factor = dot_product_cpu(dangling_bitmap, pr, V); //TODO GPU
        ////alpha choose next link, pr_tmp, beta is a priori probability (?????) that next chosen is dangling over V
        axpb_personalized_cpu(alpha, pr_tmp, alpha * dangling_factor / V, personalization_vertex, pr_tmp, V); //TODO GPU

        // Check convergence;
        double err = euclidean_distance_cpu(pr, pr_tmp, V); //TODO GPU
        converged = err <= convergence_threshold; // ????*/

        err = cudaMemcpy(val, val_gpu, V_size ,cudaMemcpyDeviceToHost);
       /* if (err != cudaSuccess) {
            fprintf(stderr,
                    "Failed to copy IN PAGERANK GPU SUPPORT from device to host (error code %s)!\n",
                    cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }*/
     /**   std::ostringstream out;
        out.precision(3);
        std::cout << "[";
        for (int i = 0; i < std::min(20, V); i++) {
            std::cout << val[i] << ", ";
        }
        std::cout << "...]";*/

        // Update the PageRank vector;

      /*  err = cudaMemcpy(pr_gpu, pr_tmp_gpu, V_size ,cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "Failed to copy IN PAGERANK GPU SUPPORT from device to host (error code %s)!\n",
                    cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }*/
        err = cudaMemcpy(pr, pr_gpu, V_size ,cudaMemcpyDeviceToHost);
        /*if (err != cudaSuccess) {
            fprintf(stderr,
                    "Failed to copy IN PAGERANK GPU SUPPORT from device to host (error code %s)!\n",
                    cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }*/
        iter++;
    }




    //free(pr_tmp);
}
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
        outdegree[y[i]]++;
    }
    // Divide each edge value by the outdegree of the source vertex;
    for (int i = 0; i < E; i++) {
    //// each node val is dependent on the number of outgoing edges
        val[i] = 1.0 / outdegree[y[i]];
    }
    free(outdegree);
}

//////////////////////////////
//////////////////////////////

// Allocate data on the CPU and GPU;
void PersonalizedPageRank::alloc() {
    // Load the input graph and preprocess it;
    initialize_graph();

    // Allocate any GPU data here;
    // TODO!

    // Size of allocations
    V_size = V * sizeof(int);
    E_size = E * sizeof(double);



    // Allocate space in VRAM
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&x_gpu, V_size);
    cudaMalloc((void **)&y_gpu, V_size);
    cudaMalloc((void **)&val_gpu, E_size);
    cudaMalloc((void **)&pr_gpu, E_size);
    cudaMalloc((void **)&pr_tmp_gpu, E_size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vectors (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Initialize data;
void PersonalizedPageRank::init() {
    // Do any additional CPU or GPU setup here;
    // TODO!

    std::cout << "INITIALIZING";
    /*cudaError_t err = cudaSuccess;
    err = cudaMemcpy(x_gpu, &x, V_size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_gpu, &y, V_size, cudaMemcpyHostToDevice);
    cudaMemcpy(val_gpu, &val, E_size, cudaMemcpyHostToDevice);
    cudaMemcpy(pr_gpu, &pr, E_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to copy IN INIT from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }*/
}

// Reset the state of the computation after every iteration.
// Reset the result, and transfer data to the GPU if necessary;
void PersonalizedPageRank::reset() {
    // Reset the PageRank vector (uniform initialization, 1 / V for each vertex);
    std::fill(pr.begin(), pr.end(), 1.0 / V);
    // Generate a new personalization vertex for this iteration;
    personalization_vertex = rand() % V;
    if (debug) std::cout << "personalization vertex=" << personalization_vertex << std::endl;
    std::cout << "RESET";

    x_array = &x[0];
    y_array = &y[0];
    pr_array = &pr[0];
    //todo val is missing
    // Do any GPU reset here, and also transfer data to the GPU;
    // TODO!
    cudaError_t err = cudaSuccess;
    cudaMemcpy(x_gpu, x_array, V_size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_gpu, y_array, V_size, cudaMemcpyHostToDevice);
    cudaMemcpy(val_gpu, &val, E_size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(pr_gpu, pr_array, E_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to copy IN RESET from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Copy data from RAM to VRAM




}

void PersonalizedPageRank::execute(int iter) {
    // Do the GPU computation here, and also transfer results to the CPU;
    //TODO! (and save the GPU PPR values into the "pr" array)
    cudaMemset(pr_gpu, 1.0 / V, E_size);  // ???
    std::cout << "EXECUTE";

    personalized_pagerank_gpu_support(V, E, dangling.data(), personalization_vertex, pr.data(), val.data(), alpha, 1e-6, 100);



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
    cuda_hello<<<1,1>>>();

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
    // TODO!
    cudaFree(x_gpu);
    cudaFree(y_gpu);
    cudaFree(val_gpu);
    cudaFree(pr_gpu);
    cudaFree(pr_tmp_gpu);
}
