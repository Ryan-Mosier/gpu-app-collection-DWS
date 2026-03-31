#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <chrono>

/**
 * CUDA Error Checking Macro
 */
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << (int)err << " (" << cudaGetErrorString(err) << ") \"" << func << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * Kernel to test warp divergence with configurable patterns.
 *
 * @param data Array to read/write results to ensure work isn't optimized away.
 * @param pattern Divergence pattern selector.
 * @param n Parameter for EVERY_N or random-like patterns.
 * @param num_elements Total number of threads/elements.
 * @param iterations Number of times to loop inside the kernel to amplify divergence impact.
 */
__global__ void synthetic_branch_kernel(int* data, int pattern, int n, int num_elements, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // Load initial value
    int val = data[idx] + idx;

    for (int i = 0; i < iterations; ++i) {
        bool branch = false;

        // Pattern logic determines the branching behavior within a warp (32 threads)
        switch (pattern) {
            case 0: // EVERY_N: Every N-th thread takes the branch
                branch = (idx % n == 0);
                break;
            case 1: // ODD_EVEN: Alternating threads (worst case for SIMT if warp size is even)
                branch = (idx % 2 == 0);
                break;
            case 2: // WARP_SPLIT: First half of warp takes one path, second half takes another
                branch = ((idx % 32) < 16);
                break;
            case 3: // INTERLEAVED_BLOCKS: Blocks of 4 threads taking the branch
                branch = ((idx / 4) % 2 == 0);
                break;
            case 4: // PSEUDO_RANDOM: Divergence changes based on iteration and index
                // A simple hash-like function to produce pseudo-random divergence
                branch = (((idx ^ i) * 0x45d9f3b) % n == 0);
                break;
            case 5: // NESTED_DIVERGENCE: Multiple levels of nested if-else
                if (idx % 2 == 0) {
                    if (idx % 4 == 0) {
                        val = (val + i) * 3;
                    } else {
                        val = (val - i) * 2;
                    }
                } else {
                    if (idx % 3 == 0) {
                        val = (val + i) * 5;
                    } else {
                        val = (val - i) * 4;
                    }
                }
                // Skip the standard branch block below for pattern 5
                continue;
            default:
                branch = (idx % 2 == 0);
                break;
        }

        // Standard branch paths for most patterns
        if (branch) {
            // Path A: Arbitrary math to simulate work
            val = (val + i) * 7;
            val ^= 0x5A5A5A5A;
            val += (idx & 0xFF);
        } else {
            // Path B: Different math to ensure SIMT divergence
            val = (val - i) * 11;
            val ^= 0xA5A5A5A5;
            val -= (idx & 0xFF);
        }
    }

    // Write back result
    data[idx] = val;
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -p <pattern>    Divergence pattern:\n"
              << "                    0: EVERY_N (N-th thread branches)\n"
              << "                    1: ODD_EVEN (T0, T2... branch)\n"
              << "                    2: WARP_SPLIT (T0-15 branch, T16-31 not)\n"
              << "                    3: INTERLEAVED (4-thread blocks branch)\n"
              << "                    4: PSEUDO_RANDOM (Dynamic divergence)\n"
              << "                    5: NESTED (Nested if-else branches)\n"
              << "  -n <value>      Value of N for EVERY_N or RANDOM patterns (default: 2)\n"
              << "  -t <threads>    Total number of threads (default: 65536)\n"
              << "  -i <iters>      Kernel loop iterations (default: 100)\n"
              << "  -b <blocksize>  Threads per block (default: 128)\n"
              << "  -h              Show this help\n";
}

int main(int argc, char** argv) {
    int pattern = 1;      // Default: ODD_EVEN
    int n = 2;            // Default: N=2
    int num_elements = 65536;
    int iterations = 100;
    int block_size = 128;

    // Simple manual argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-p" && i + 1 < argc) pattern = std::stoi(argv[++i]);
        else if (arg == "-n" && i + 1 < argc) n = std::stoi(argv[++i]);
        else if (arg == "-t" && i + 1 < argc) num_elements = std::stoi(argv[++i]);
        else if (arg == "-i" && i + 1 < argc) iterations = std::stoi(argv[++i]);
        else if (arg == "-b" && i + 1 < argc) block_size = std::stoi(argv[++i]);
        else if (arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "--- Synthetic Branch Benchmark ---\n"
              << "Pattern:    " << pattern << "\n"
              << "N:          " << n << "\n"
              << "Threads:    " << num_elements << "\n"
              << "Iterations: " << iterations << "\n"
              << "Block Size: " << block_size << "\n"
              << "----------------------------------\n";

    int* d_data;
    size_t size = num_elements * sizeof(int);

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, size));

    // Initialize host data
    std::vector<int> h_data(num_elements, 12345);
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice));

    int grid_size = (num_elements + block_size - 1) / block_size;

    // Kernel timing using CUDA events for better accuracy
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    synthetic_branch_kernel<<<grid_size, block_size>>>(d_data, pattern, n, num_elements, iterations);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));

    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_data.data(), d_data, size, cudaMemcpyDeviceToHost));

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    // Output sample results to verify execution
    std::cout << "Sample output [0-7]: ";
    for (int i = 0; i < std::min(num_elements, 8); ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << "\nBenchmark completed successfully." << std::endl;

    return 0;
}
