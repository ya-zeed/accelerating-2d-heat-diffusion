#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

// Define constants
#define A 110.0
#define LENGTH 1000.0
#define TIME 80.0
#define NODES 800

// Function to initialize the temperature matrix and set boundary conditions
void initialize(double *u, int nodes) {
    int i, j;
    double step = 100.0 / (nodes - 1);

    // Set boundary conditions
    for (i = 0; i < nodes; i++) {
        double boundary_value = step * i;
        for (j = 0; j < nodes; j++) {
            // Top and bottom boundaries
            u[i * nodes + j] = 20.0; // Default initialization
            if (i == 0 || i == nodes - 1) u[i * nodes + j] = boundary_value;

            // Left and right boundaries
            if (j == 0 || j == nodes - 1) u[i * nodes + j] = boundary_value;
        }
    }
}

// Function to copy a matrix
void copy_matrix(double *dest, double *src, int size) {
    memcpy(dest, src, size * sizeof(double));
}

// Main simulation function
void simulate(double *u, int nodes, double dx, double dy, double dt, double a, int t_nodes) {
    int i, j, t;
    double *w = (double *)malloc(nodes * nodes * sizeof(double));

    for (t = 0; t < t_nodes; t++) {
        copy_matrix(w, u, nodes * nodes);

        // Perform computation
        for (i = 1; i < nodes - 1; i++) {
            for (j = 1; j < nodes - 1; j++) {
                double dd_ux = (w[(i - 1) * nodes + j] - 2.0 * w[i * nodes + j] + w[(i + 1) * nodes + j]) / (dx * dx);
                double dd_uy = (w[i * nodes + (j - 1)] - 2.0 * w[i * nodes + j] + w[i * nodes + (j + 1)]) / (dy * dy);

                u[i * nodes + j] = dt * a * (dd_ux + dd_uy) + w[i * nodes + j];
            }
        }
    }

    free(w);
}

int main() {
    int nodes = NODES;
    double length = LENGTH;
    double time = TIME;
    double dx = length / (nodes - 1);
    double dy = length / (nodes - 1);
    double dt = fmin(dx * dx / (4.0 * A), dy * dy / (4.0 * A));
    int t_nodes = (int)(time / dt) + 1;

    // Allocate memory for the temperature grid
    double *u = (double *)malloc(nodes * nodes * sizeof(double));

    // Initialize temperature grid and set boundary conditions
    initialize(u, nodes);

    // Start timer
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Run simulation
    simulate(u, nodes, dx, dy, dt, A, t_nodes);

    // Stop timer
    gettimeofday(&end, NULL);

    // Calculate elapsed time in seconds
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Simulation completed in %f seconds.\n", elapsed);


    // Free memory
    free(u);

    return 0;
}
