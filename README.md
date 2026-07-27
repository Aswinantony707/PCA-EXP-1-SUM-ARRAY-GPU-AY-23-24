# PCA: EXP-1  SUM ARRAY GPU
<h3>ENTER YOUR NAME</h3>
<h3>ENTER YOUR REGISTER NO</h3>
<h3>EX. NO</h3>
<h3>DATE</h3>
<h1> <align=center> SUM ARRAY ON HOST AND DEVICE </h3>
PCA-GPU-based-vector-summation.-Explore-the-differences.
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution configuration of block.x = 1024. Try to explain the difference and the reason.

ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution confi gurations.
## AIM:

To perform vector addition on host and device.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1. Initialize the device and set the device properties.
2. Allocate memory on the host for input and output arrays.
3. Initialize input arrays with random values on the host.
4. Allocate memory on the device for input and output arrays, and copy input data from host to device.
5. Launch a CUDA kernel to perform vector addition on the device.
6. Copy output data from the device to the host and verify the results against the host's sequential vector addition. Free memory on the host and the device.

## PROGRAM:
```
%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = true;

    for (int i = 0; i < N; i++)
    {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = false;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n",
                   hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
}

void initialData(float *ip, int size)
{
    srand((unsigned)time(NULL));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}

// CUDA Kernel
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // Vector size
    int nElem = 1<<24;
    printf("Vector size %d\n", nElem);

    size_t nBytes = nElem * sizeof(float);

    // Host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    double iStart, iElaps;

    // Initialize data
    iStart = seconds();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = seconds() - iStart;
    printf("initialData Time elapsed %f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // CPU computation
    iStart = seconds();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    iElaps = seconds() - iStart;
    printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);

    // Device memory
    float *d_A, *d_B, *d_C;

    CHECK(cudaMalloc((float **)&d_A, nBytes));
    CHECK(cudaMalloc((float **)&d_B, nBytes));
    CHECK(cudaMalloc((float **)&d_C, nBytes));

    // Copy host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

    // Kernel configuration
    int blockSize = 1024;
    dim3 block(blockSize);
    dim3 grid((nElem + block.x - 1) / block.x);

    // Launch kernel
    iStart = seconds();
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    printf("sumArraysOnGPU <<<%d, %d>>> Time elapsed %f sec\n",
           grid.x, block.x, iElaps);

    CHECK(cudaGetLastError());

    // Copy result back
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // Verify
    checkResult(hostRef, gpuRef, nElem);

    // Free device memory<img width="1232" height="247" alt="Screenshot 2026-07-27 094647" src="https://github.com/user-attachments/assets/0726ba6d-da66-461c-b95b-d22010c40541" />
<img width="1231" height="248" alt="Screenshot 2026-07-27 094720" src="https://github.com/user-attachments/assets/5bd675c5-bc7b-460d-9fb6-a7be362ceb7c" />
<img width="1232" height="257" alt="Screenshot 2026-07-27 094619" src="https://github.com/user-attachments/assets/5cb537ec-c5c3-4fcc-b0e4-cf785dfd7398" />

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}
```
## OUTPUT:
<img width="1231" height="248" alt="Screenshot 2026-07-27 094720" src="https://github.com/user-attachments/assets/25da4b84-250c-49b9-992b-04c366c9cb00" />

<img width="1232" height="247" alt="Screenshot 2026-07-27 094647" src="https://github.com/user-attachments/assets/f2cb0695-5d46-45d5-a893-d4a230631ed2" />

<img width="1232" height="257" alt="Screenshot 2026-07-27 094619" src="https://github.com/user-attachments/assets/a2b84aba-c9be-4b89-9409-ee08238d9b72" />


## RESULT:
Thus, Implementation of sum arrays on host and device is done in nvcc cuda using random number.
