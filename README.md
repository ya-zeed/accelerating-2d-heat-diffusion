# Accelerating a 2D Heat Diffusion Simulation

This project explores the acceleration of a 2D heat diffusion simulation using various parallel computing techniques, including OpenACC, CUDA Python, and CUDA C/C++. The aim is to achieve significant performance improvements over a baseline CPU implementation.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup and Usage](#setup-and-usage)
- [Performance Comparison](#performance-comparison)
- [Contributors](#contributors)

---

## Overview

Heat diffusion simulations are widely used in fields like engineering, climate modeling, and materials science. These simulations involve solving partial differential equations (PDEs) that can be computationally expensive, especially for large domains. This project demonstrates how GPU acceleration can significantly improve the runtime of such simulations.

Key Features:
- **Baseline CPU Implementation**: Implemented in both Python and C.
- **GPU Accelerated Implementations**:
  - OpenACC (C with minimal pragmas for GPU parallelization).
  - CUDA Python using Numba.
  - CUDA C/C++ with optimized kernel functions.
- **Comparative Analysis**: Performance, ease of use, and maintainability.

---

## Project Structure

```
├── Final Report.pdf             # Comprehensive report describing the project methodology and results
├── cpu_version.c                # Baseline CPU implementation in C
├── openAcc_version.c            # OpenACC implementation in C
├── cuda_c_version.cu            # High-performance CUDA C/C++ implementation
├── numba.ipynb                  # CUDA Python implementation using Numba
└── README.md                    # This README file
```

---

## Setup and Usage

### Prerequisites
- A system with a CUDA-capable GPU.
- Required software:
  - NVIDIA CUDA Toolkit
  - GCC Compiler
  - Python with `Numba` installed (for the CUDA Python version)

### Compilation and Execution

#### 1. CPU Implementation
```bash
gcc -O3 cpu_version.c -o cpu_version
./cpu_version
```

#### 2. OpenACC Implementation
```bash
nvc -acc -fast openAcc_version.c -o openAcc_version
./openAcc_version
```

#### 3. CUDA C/C++ Implementation
```bash
nvcc cuda_c_version.cu -o cuda_c_version
./cuda_c_version
```

#### 4. CUDA Python Implementation
Run the `numba.ipynb` notebook in Jupyter or execute the Python script directly.

---

## Performance Comparison

The following table highlights the performance gains achieved using different GPU programming techniques.

| Length (m) | Nodes | CPU Time (s) | OpenACC (s) | CUDA Python (s) | CUDA C (s) |
|------------|-------|--------------|-------------|-----------------|------------|
| 500        | 400   | 3960         | 0.40        | 1.64            | 0.20       |
| 1000       | 800   | 22.7         | 1.175       | 3.61            | 0.91       |
| 2000       | 1600  | 207.0        | 6.18        | 10.7            | 5.80       |

**Key Insights**:
- **CUDA C/C++**: Offers the best performance but requires significant programming effort.
- **OpenACC**: A good balance of performance and simplicity.
- **CUDA Python**: Easy to implement, but slightly slower compared to CUDA C/C++.

---

## Contributors

- **Yazeed Altaweel** (201812960)
- **Malek Aljemaili** (201431340)
- **Yousef Alzahrani** (201678960)

This project was completed as part of COE-506 - GPU Programming & Architecture at King Fahd University of Petroleum & Minerals.
