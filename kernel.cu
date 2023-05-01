
#include "cuda_runtime.h"
#include "cuda_occupancy.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

#include <stdio.h>
#include <cstdint>
#include <cmath>
#include <ctime>

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory {
    __device__ inline operator T* () {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T* () const {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <>
struct SharedMemory<double> {
    __device__ inline operator double* () {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }

    __device__ inline operator const double* () const {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }
};

template <class T>
__device__ __forceinline__ T warpReduceSum(unsigned int mask, T mySum) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        mySum += __shfl_down_sync(mask, mySum, offset);
    }
    return mySum;
}

#if __CUDA_ARCH__ >= 800
// Specialize warpReduceFunc for int inputs to use __reduce_add_sync intrinsic
// when on SM 8.0 or higher
template <>
__device__ __forceinline__ int warpReduceSum<int>(unsigned int mask,
    int mySum) {
    mySum = __reduce_add_sync(mask, mySum);
    return mySum;
}
#endif

cudaError_t CudaReduce(int *c, const int *a, unsigned int size);
cudaError_t CudaVectorSum(int* c, const int* a, unsigned int size);

//template<class Type>
//cudaError_t CudaAccumulate(Type* c, const Type* a, unsigned int size);

cudaError_t CudaAccumulate(uint32_t* arg_pAccumulate, const uint32_t* arg_pVector, unsigned int arg_dSize);
cudaError_t CudaSparseToDense(uint32_t* arg_pArrayToModify, uint32_t arg_dSize);


__global__ void KernelAccumulateArray(int *c, int *a)
{

}

template <class T>
__global__ void reduce0(T* g_idata, T* g_odata, unsigned int n) {
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T* sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        // modulo arithmetic is slow!
        if ((tid % (2 * s)) == 0) {
            sdata[tid] += sdata[tid + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <class T>
__global__ void reduce3(T* g_idata, T* g_odata, unsigned int n) {
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T* sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    T mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockDim.x < n) mySum += g_idata[i + blockDim.x];

    sdata[tid] = mySum;
    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

template <typename T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce7(const T* __restrict__ g_idata, T* __restrict__ g_odata,
    unsigned int n) {
    T* sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = blockSize * gridDim.x;
    unsigned int maskLength = (blockSize & 31);  // 31 = warpSize-1
    maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
    const unsigned int mask = (0xffffffff) >> maskLength;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    if (nIsPow2) {
        unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
        gridSize = gridSize << 1;

        while (i < n) {
            mySum += g_idata[i];
            // ensure we don't read out of bounds -- this is optimized away for
            // powerOf2 sized arrays
            if ((i + blockSize) < n) {
                mySum += g_idata[i + blockSize];
            }
            i += gridSize;
        }
    }
    else {
        unsigned int i = blockIdx.x * blockSize + threadIdx.x;
        while (i < n) {
            mySum += g_idata[i];
            i += gridSize;
        }
    }

    // Reduce within warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
    // SM 8.0
    mySum = warpReduceSum<T>(mask, mySum);

    // each thread puts its local sum into shared memory
    if ((tid % warpSize) == 0) {
        sdata[tid / warpSize] = mySum;
    }

    __syncthreads();

    const unsigned int shmem_extent =
        (blockSize / warpSize) > 0 ? (blockSize / warpSize) : 1;
    const unsigned int ballot_result = __ballot_sync(mask, tid < shmem_extent);
    if (tid < shmem_extent) {
        mySum = sdata[tid];
        // Reduce final warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
        // SM 8.0
        mySum = warpReduceSum<T>(ballot_result, mySum);
    }

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = mySum;
    }
}

// Performs a reduction step and updates numTotal with how many are remaining
template <typename T, typename Group>
__device__ T cg_reduce_n(T in, Group& threads) {
    return cg::reduce(threads, in, cg::plus<T>());
}

template <class T>
__global__ void cg_reduce(T* g_idata, T* g_odata, unsigned int n) {
    // Shared memory for intermediate steps
    T* sdata = SharedMemory<T>();
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    // Handle to tile in thread block
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

    unsigned int ctaSize = cta.size();
    unsigned int numCtas = gridDim.x;
    unsigned int threadRank = cta.thread_rank();
    unsigned int threadIndex = (blockIdx.x * ctaSize) + threadRank;

    T threadVal = 0;
    {
        unsigned int i = threadIndex;
        unsigned int indexStride = (numCtas * ctaSize);
        while (i < n) {
            threadVal += g_idata[i];
            i += indexStride;
        }
        sdata[threadRank] = threadVal;
    }

    // Wait for all tiles to finish and reduce within CTA
    {
        unsigned int ctaSteps = tile.meta_group_size();
        unsigned int ctaIndex = ctaSize >> 1;
        while (ctaIndex >= 32) {
            cta.sync();
            if (threadRank < ctaIndex) {
                threadVal += sdata[threadRank + ctaIndex];
                sdata[threadRank] = threadVal;
            }
            ctaSteps >>= 1;
            ctaIndex >>= 1;
        }
    }

    // Shuffle redux instead of smem redux
    {
        cta.sync();
        if (tile.meta_group_rank() == 0) {
            threadVal = cg_reduce_n(threadVal, tile);
        }
    }

    if (threadRank == 0) g_odata[blockIdx.x] = threadVal;
}

template <class T>
__global__ void vector_sum(T* g_idata, T* g_odata, unsigned int n) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    uint32_t threadId = threadIdx.x;
    uint32_t threadCount = gridDim.x;

    if (threadId < n) {
        uint32_t temp = g_odata[threadId] + g_idata[threadId];
        g_odata[threadId] = temp;
    }
}

template <class Type>
__global__ void masked_array_to_list(Type* arg_array, Type arg_data_zero, uint32_t arg_array_size, uint32_t arg_bin_start_width = 0)
{
    Type* pCommonMem = SharedMemory<Type>();

    uint32_t kernelThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t kernelStride = blockDim.x * gridDim.x;

    uint32_t threadId = threadIdx.x;
    uint32_t threadCount = blockDim.x;

    if (kernelThreadId < arg_array_size) {
        pCommonMem[threadId] = arg_array[kernelThreadId];
    }

    __syncthreads();

    for (uint32_t loop_stride = 2; loop_stride <= threadCount; loop_stride <<= 1) {
        if (threadId % loop_stride == 0) {
            uint32_t idx_left = threadId, idx_right = threadId + loop_stride - 1;

            while (idx_right > idx_left) {
                if (pCommonMem[idx_left] == arg_data_zero) {
                    if (pCommonMem[idx_right] != arg_data_zero) {
                        Type tempValue = pCommonMem[idx_left];
                        pCommonMem[idx_left] = pCommonMem[idx_right];
                        pCommonMem[idx_right] = tempValue;
                    }
                    else {
                        idx_right--;
                    }
                }
                else {
                    idx_left++;
                }
            }
        }

        __syncthreads();
    }

    if (kernelThreadId < arg_array_size) {
        arg_array[kernelThreadId] = pCommonMem[threadId];
    }
}

typedef void (*SimpleFunc)(int* c, int* a);

int main()
{
    const unsigned int array_size = 2000;
    const unsigned int count = 200;

    const unsigned int sparse_array_size = 128;
    const unsigned int sparse_data_count = 48;

    unsigned int sparse_a[array_size] = {};

    unsigned int natural_a[array_size] = {};
    
    unsigned int sum_array[array_size] = {};

    unsigned int acc_array[array_size] = {};

    unsigned int sparse_array[sparse_array_size] = {};

    srand(time(0));

    int gridSize, blockSize;

    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, &vector_sum<uint32_t>, 0, 0);

    //Populate the sparse
     for (uint32_t sparse_elem_count = 0; sparse_elem_count < count; sparse_elem_count++) {
        uint32_t idx_sparse_elem = rand() % array_size;

        while (sparse_a[idx_sparse_elem] == 1) {
            idx_sparse_elem++;
            idx_sparse_elem %= array_size;
        }

        sparse_a[idx_sparse_elem] = 1;
    }

    //Populate the sparse
    for (uint32_t sparse_elem_count = 0; sparse_elem_count < sparse_data_count; sparse_elem_count++) {
        uint32_t idx_sparse_elem = rand() % sparse_array_size;

        while (sparse_array[idx_sparse_elem] == 1) {
            idx_sparse_elem++;
            idx_sparse_elem %= sparse_array_size;
        }

        sparse_array[idx_sparse_elem] = 1;
    }

    //Populate Natural numbers
    for (uint32_t idx_nat = 0; idx_nat < 1024; idx_nat++) {
        natural_a[idx_nat] = idx_nat + 1;
    }

    //Check
    uint32_t sum_check = 0;
    for (uint32_t idx = 0; idx < array_size; idx++) {
        if (sparse_a[idx] == 1) sum_check++;
    }

    printf("Check: %d\n", sum_check);

    // Add vectors in parallel.
    cudaError_t cudaStatus = CudaAccumulate(sum_array, sparse_a, array_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaReduce failed!");
        return 1;
    }

    // Add vectors in parallel.
    //cudaStatus = CudaVectorSum(acc_array, natural_a, 1024);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "CudaReduce failed!");
    //    return 1;
    //}

    printf("{%d,%d,%d,%d,%d}\n",
        sum_array[0], sum_array[1], sum_array[2], sum_array[3], sum_array[4]);

    cudaStatus = CudaSparseToDense(sparse_array, sparse_array_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaSparseToDense failed!");
        return 1;
    }

    printf("{%d,%d,%d,%d,%d}\n",
        sparse_array[0], sparse_array[1], sparse_array[2], sparse_array[3], sparse_array[4]);

    // Add vectors in parallel.
    //cudaStatus = CudaVectorSum(acc_array, natural_a, 1024);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "CudaReduce failed!");
    //    return 1;
    //}

    printf("{%d,%d,%d,%d,%d}\n",
        acc_array[0], acc_array[1], acc_array[2], acc_array[3], acc_array[4]);

    printf("{%d,%d,%d,%d,%d}\n",
        acc_array[30], acc_array[31], acc_array[32], acc_array[33], acc_array[34]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t CudaReduce(int* c, const int* a, unsigned int size)
{
    int *dev_a = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_c, c, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    uint32_t smemSize = 1024 * sizeof(int);
    reduce0<int> <<<1, 1024, smemSize>>> (dev_a, dev_c, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    
    return cudaStatus;
}

cudaError_t CudaVectorSum(int* c, const int* a, unsigned int size)
{
    int* dev_a = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_c, c, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    // Launch a kernel on the GPU with one thread for each element.
    uint32_t smemSize = 1024 * sizeof(int);
    vector_sum<int> <<<1, 1024>>> (dev_a, dev_c, 1024);
    vector_sum<int> <<<1, 16>>> (dev_a + 32, dev_c + 32, 16);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);

    return cudaStatus;
}

cudaError_t CudaAccumulate(uint32_t* arg_pAccumulate, const uint32_t* arg_pVector, unsigned int arg_dSize)
{
    cudaError_t cudaStatus = cudaError::cudaSuccess;

    uint32_t* dev_pVector = nullptr;
    uint32_t* dev_pAccumulate = nullptr;

    uint32_t dSliceSize = 32;
    uint32_t dSliceCount = 0;

    if (!arg_pVector || !arg_pAccumulate || !arg_dSize) return cudaError::cudaErrorAssert;

    if (arg_dSize > dSliceSize) {
        for (; dSliceSize < 512; dSliceSize *= 2)
            if (arg_dSize / dSliceSize == 0 && arg_dSize / (dSliceSize * 2) == 1)
                break;
    }

    uint32_t dRemainder = arg_dSize % dSliceSize;
    if (dRemainder == 0) {
        dSliceCount = arg_dSize / dSliceSize;
    }
    else {
        dSliceCount = (arg_dSize / dSliceSize) + 1;
    }

    uint32_t smemSize = dSliceSize * sizeof(uint32_t);
    uint32_t dBufferSize = dSliceCount * dSliceSize;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_pVector, dBufferSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_pAccumulate, dBufferSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemset(dev_pVector, 0, dBufferSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_pVector, arg_pVector, arg_dSize * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemset(dev_pAccumulate, 0, dBufferSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    for (uint32_t iter_slice = 0; iter_slice < dSliceCount; iter_slice++) {
        cg_reduce<uint32_t> <<<1, dSliceSize, smemSize>>> (dev_pVector + (iter_slice * dSliceSize), dev_pAccumulate + (iter_slice * dSliceSize), dSliceSize);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cg_reduce launch failed: %s in iteration %d\n", cudaGetErrorString(cudaStatus), iter_slice);
            goto Error;
        }

        if (iter_slice != 0) {
            vector_sum<uint32_t> <<<1, dSliceSize>>> (dev_pAccumulate + (iter_slice * dSliceSize), dev_pAccumulate, dSliceSize);

            // Check for any errors launching the kernel
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "vector_sum launch failed: %s in iteration %d\n", cudaGetErrorString(cudaStatus), iter_slice);
                goto Error;
            }
        }
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(arg_pAccumulate, dev_pAccumulate, arg_dSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_pVector);
    cudaFree(dev_pAccumulate);

    return cudaStatus;

}

cudaError_t CudaSparseToDense(uint32_t* arg_pArrayToModify, uint32_t arg_dSize)
{
    uint32_t* dev_a = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for array
    cudaStatus = cudaMalloc((void**)&dev_a, arg_dSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, arg_pArrayToModify, arg_dSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    uint32_t smemSize = arg_dSize * sizeof(int);
    masked_array_to_list<uint32_t> 
        <<<1, 128, smemSize>>>
        (dev_a, 0u, arg_dSize);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(arg_pArrayToModify, dev_a, arg_dSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_a);

    return cudaStatus;
}