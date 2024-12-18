#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cuda.h>

// Define constants
#define A 110.0
#define LENGTH 1000.0
#define TIME 80.0
#define NODES 800

// Initialize the temperature matrix and set boundary conditions on host
void initialize(double *u, int nodes) {
    int i, j;
    double step = 100.0 / (nodes - 1);

    // Set boundary conditions
    for (i = 0; i < nodes; i++) {
        double boundary_value = step * i;
        for (j = 0; j < nodes; j++) {
            u[i * nodes + j] = 20.0; // Default initialization
            if (i == 0 || i == nodes - 1) u[i * nodes + j] = boundary_value;
            if (j == 0 || j == nodes - 1) u[i * nodes + j] = boundary_value;
        }
    }
}

// Kernel to copy u into w
__global__ void copy_kernel(double *u, double *w, int nodes) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nodes && j < nodes) {
        w[i * nodes + j] = u[i * nodes + j];
    }
}

// Kernel to perform one time-step of the finite difference computation
__global__ void step_kernel(double *u, double *w, int nodes, double dx, double dy, double dt, double a) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < nodes - 1 && j > 0 && j < nodes - 1) {
        double dd_ux = (w[(i - 1) * nodes + j] - 2.0 * w[i * nodes + j] + w[(i + 1) * nodes + j]) / (dx * dx);
        double dd_uy = (w[i * nodes + (j - 1)] - 2.0 * w[i * nodes + j] + w[i * nodes + (j + 1)]) / (dy * dy);
        u[i * nodes + j] = dt * a * (dd_ux + dd_uy) + w[i * nodes + j];
    }
}

int main() {
    int nodes = NODES;
    double length = LENGTH;
    double time = TIME;
    double dx = length / (nodes - 1);
    double dy = length / (nodes - 1);
    double dt = fmin(dx * dx / (4.0 * A), dy * dy / (4.0 * A));
    int t_nodes = (int)(time / dt) + 1;

    // Allocate memory on host
    double *h_u = (double *)malloc(nodes * nodes * sizeof(double));

    // Initialize on host
    initialize(h_u, nodes);

    // Allocate memory on device
    double *d_u, *d_w;
    cudaMalloc((void**)&d_u, nodes * nodes * sizeof(double));
    cudaMalloc((void**)&d_w, nodes * nodes * sizeof(double));

    // Copy u to device
    cudaMemcpy(d_u, h_u, nodes * nodes * sizeof(double), cudaMemcpyHostToDevice);

    // Set up CUDA grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((nodes + blockDim.x - 1) / blockDim.x, (nodes + blockDim.y - 1) / blockDim.y);

    // Start timer
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Time-stepping loop on host
    for (int t = 0; t < t_nodes; t++) {
        // Copy u into w
        copy_kernel<<<gridDim, blockDim>>>(d_u, d_w, nodes);

        // Compute the next timestep
        step_kernel<<<gridDim, blockDim>>>(d_u, d_w, nodes, dx, dy, dt, A);

        // Synchronize to ensure kernels have completed
        cudaDeviceSynchronize();
    }

    // Stop timer
    gettimeofday(&end, NULL);

    // Copy result back to host
    cudaMemcpy(h_u, d_u, nodes * nodes * sizeof(double), cudaMemcpyDeviceToHost);

    // Calculate elapsed time in seconds
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Simulation completed in %f seconds.\n", elapsed);

    // Free memory
    free(h_u);
    cudaFree(d_u);
    cudaFree(d_w);

    return 0;
}
