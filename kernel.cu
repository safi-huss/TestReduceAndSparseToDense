
#include "cuda_runtime.h"
#include "cuda_occupancy.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <assert.h>

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

cudaError_t CudaReduce(unsigned int *c, const unsigned int *a, unsigned int size, const unsigned int arg_dSliceSize);
cudaError_t CudaVectorSum(int* c, const int* a, unsigned int size);

//template<class Type>
//cudaError_t CudaAccumulate(Type* c, const Type* a, unsigned int size);

cudaError_t CudaAccumulate(uint32_t* arg_pAccumulate, const uint32_t* arg_pVector, unsigned int arg_dSize, const unsigned int arg_dSliceSize);
cudaError_t CudaSparseToPackedCell(uint32_t* arg_pArrayToModify, uint32_t arg_dSize, const uint32_t arg_dSliceSize);
cudaError_t CudaMergeCells(uint32_t* arg_pArrayToModify, uint32_t* arg_pCellLengthArray, uint32_t arg_dSize, const uint32_t arg_dCellSize, const uint32_t arg_dCellCount);

cudaError_t CudaAccumulateAndPack(uint32_t* arg_pArray, const uint32_t arg_dSize);


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
__global__ void masked_array_to_list_first(Type* arg_array, Type arg_data_zero, uint32_t arg_array_size)
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

/* arg_length_array_size == blockDom.x // the number of length elements passed should be equal to the number of threads */
template <class Type>
__global__ void masked_array_to_list_second(Type* arg_array, Type* arg_length_array, const uint32_t arg_array_size, const uint32_t arg_length_array_size, const uint32_t arg_cell_size)
{
    uint32_t kernelThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t kernelStride = blockDim.x * gridDim.x;

    uint32_t threadId = threadIdx.x;
    uint32_t threadCount = blockDim.x;

    for (uint32_t loop_stride = 2; loop_stride <= arg_length_array_size; loop_stride <<= 1) {
        if (threadId % loop_stride == 0) {
            uint32_t left_length = arg_length_array[threadId], right_length = arg_length_array[threadId + loop_stride - (loop_stride >> 1)];
            uint32_t left_start = arg_cell_size * threadId, right_start = arg_cell_size * threadId + (arg_cell_size * (loop_stride >> 1));
            uint32_t transfered_elems = 0;

            while (transfered_elems < right_length) {
                Type temp = arg_array[left_start + left_length + transfered_elems];
                arg_array[left_start + left_length + transfered_elems] = arg_array[right_start + transfered_elems];
                arg_array[right_start + transfered_elems] = temp;

                transfered_elems++;
            }

            arg_length_array[threadId] += arg_length_array[threadId + loop_stride - (loop_stride >> 1)];
        }

        __syncthreads();
    }

    __syncthreads();
}

template <class Type>
__global__ void collect_reduce_first_elem(Type* arg_result, Type* arg_array, uint32_t arg_bin_size, uint32_t arg_bin_count, uint32_t arg_array_size)
{
    uint32_t kernelThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t kernelStride = blockDim.x * gridDim.x;

    uint32_t threadId = threadIdx.x;
    uint32_t threadCount = blockDim.x;

    for (uint32_t idx_bin = 0; idx_bin < arg_bin_count; idx_bin += kernelStride) {
        arg_result[idx_bin + kernelThreadId] = arg_array[idx_bin + kernelThreadId * arg_bin_size]; 
    }
}

template <class Type>
__global__ void swap(Type* arg_dest, Type* arg_src, const uint32_t arg_size)
{
    uint32_t kernelThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t kernelStride = blockDim.x * gridDim.x;

    for (uint32_t idx_stride = 0; idx_stride < arg_size; idx_stride += kernelStride) {
        Type temp = arg_dest[idx_stride];
        arg_dest[idx_stride] = arg_src[idx_stride];
        arg_src[idx_stride] = temp;
    }
}

typedef void (*SimpleFunc)(int* c, int* a);

int main()
{
    const unsigned int array_size = 4096;
    const unsigned int slice_size = 64;
    const unsigned int count = 600;

    unsigned int sparse_array[array_size] = {};
    
    unsigned int sum_array[array_size] = {};

    unsigned int acc_array[array_size] = {};

    unsigned int sum = 0;

    srand(time(0));

    //Populate the sparse
     for (uint32_t sparse_elem_count = 0; sparse_elem_count < count; sparse_elem_count++) {
        uint32_t idx_sparse_elem = rand() % array_size;

        while (sparse_array[idx_sparse_elem] == 1) {
            idx_sparse_elem++;
            idx_sparse_elem %= array_size;
        }

        sparse_array[idx_sparse_elem] = 1;
    }

    //Check
    uint32_t sum_check = 0;
    for (uint32_t idx = 0; idx < array_size; idx++) {
        if (sparse_array[idx] == 1) sum_check++;
    }

    printf("Total Sum: %d\n", sum_check);

    //Slice Sum
    for (uint32_t slice_idx = 0; slice_idx < array_size; slice_idx += slice_size) {
        uint32_t slice_sum = 0;

        for (uint32_t inner_idx = 0; inner_idx < slice_size; inner_idx++) {
            slice_sum += sparse_array[slice_idx + inner_idx];
        }

        sum_array[slice_idx / slice_size] = slice_sum;
    }

    //Print Slice Sums
    printf("Slice Sums:\n");
    for (auto idx_sums = 0; idx_sums < slice_size; idx_sums++) {
        printf("%d, ", sum_array[idx_sums]);
    }
    printf("\n\n");

    // Add vectors in parallel.
    cudaError_t cudaStatus = CudaReduce(sum_array, sparse_array, array_size, slice_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaReduce failed!");
        return 1;
    }

    //Print Slice Sums
    printf("Cuda Slice Sums:\n");
    for (auto idx_sums = 0; idx_sums < slice_size; idx_sums++) {
        printf("%d, ", sum_array[idx_sums]);
    }
    printf("\n\n");

    // Add vectors in parallel.
    cudaStatus = CudaAccumulate(&sum, sparse_array, array_size, slice_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaAccumulate failed!");
        return 1;
    }

    printf("Cuda Total Sum: %d\n\n", sum);

    cudaStatus = CudaSparseToPackedCell(sparse_array, array_size, slice_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaSparseToPackedCell failed!");
        return 1;
    }

    printf("Cuda Packed Cell Elems:\n");
    for (uint32_t slice_idx = 0; slice_idx < array_size; slice_idx += slice_size) {
        uint32_t slice_sum = 0;
        uint32_t slice_packed_len = sum_array[slice_idx / slice_size];

        for (uint32_t inner_idx = 0; inner_idx < slice_packed_len; inner_idx++) {
            slice_sum += sparse_array[slice_idx + inner_idx];
        }

        printf("%d, ", slice_sum);
    }
    printf("\n\n");

    printf("Cuda Array Pre-merge:\n");
    for (uint32_t slice_idx = 0; slice_idx < array_size; slice_idx += slice_size) {
        printf("%3d: ", sum_array[slice_idx / slice_size]);

        for (uint32_t inner_idx = 0; inner_idx < slice_size; inner_idx++) {
            printf("%d, ", sparse_array[slice_idx + inner_idx]);
        }

        printf("\n");
    }
    printf("\n\n");

    cudaStatus = CudaMergeCells(sparse_array, sum_array, array_size, slice_size, array_size / slice_size);

    uint32_t packed_count_cuda = 0;
    for (uint32_t array_idx = 0; array_idx < sum; array_idx++) {
        packed_count_cuda += sparse_array[array_idx];
    }
    printf("Cuda Fully Packed Array: %d\n\n", packed_count_cuda);

    printf("Cuda Final Array:\n");
    for (uint32_t slice_idx = 0; slice_idx < array_size; slice_idx += slice_size) {
        printf("%3d: ", sum_array[slice_idx / slice_size]);

        for (uint32_t inner_idx = 0; inner_idx < slice_size; inner_idx++) {
            printf("%d, ", sparse_array[slice_idx + inner_idx]);
        }

        printf("\n");
    }
    printf("\n\n");


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
cudaError_t CudaReduce(unsigned int* arg_pResult, const unsigned int* arg_pVector, unsigned int arg_dSize, const unsigned int arg_dSliceSize)
{
    assert(arg_dSliceSize <= 1024);

    unsigned int *dev_pVector = 0;
    unsigned int *dev_pResult = 0;
    cudaError_t cudaStatus;

    uint32_t dSliceCount = (arg_dSize + arg_dSliceSize - 1) / arg_dSliceSize;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have arg_pVector CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_pResult, dSliceCount * arg_dSliceSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_pVector, dSliceCount * arg_dSliceSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_pVector, arg_pVector, arg_dSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemset(dev_pResult, 0, arg_dSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    uint32_t smemSize = arg_dSliceSize * sizeof(int);
    reduce0<uint32_t> <<<dSliceCount, arg_dSliceSize, smemSize>>> (dev_pVector, dev_pResult, arg_dSize);

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
    cudaStatus = cudaMemcpy(arg_pResult, dev_pResult, arg_dSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_pResult);
    cudaFree(dev_pVector);
    
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
        fprintf(stderr, "cudaSetDevice failed!  Do you have arg_pVector CUDA-capable GPU installed?");
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

cudaError_t CudaAccumulate(uint32_t* arg_pAccumulate, const uint32_t* arg_pVector, unsigned int arg_dSize, const unsigned int arg_dSliceSize)
{
    assert(arg_dSliceSize <= 1024);

    cudaError_t cudaStatus = cudaError::cudaSuccess;

    uint32_t* dev_pVector = nullptr;
    uint32_t* dev_pAccumulateStage1 = nullptr;
    uint32_t* dev_pAccumulateStage2 = nullptr;

    uint32_t dSliceSize = 32;

    if (!arg_pVector || !arg_pAccumulate || !arg_dSize) return cudaError::cudaErrorAssert;

    if (arg_dSliceSize == 0) {
        if (arg_dSize > dSliceSize) {
            for (; dSliceSize < 512; dSliceSize *= 2)
                if (arg_dSize / dSliceSize == 0 && arg_dSize / (dSliceSize * 2) == 1)
                    break;
        }
    }
    else {
        dSliceSize = arg_dSliceSize;
    }

    uint32_t dSliceCount = (arg_dSize + arg_dSliceSize - 1) / arg_dSliceSize;

    assert(dSliceCount <= 1024);

    uint32_t smemSize = dSliceSize * sizeof(uint32_t);
    uint32_t dBufferSize = dSliceCount * dSliceSize;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have arg_pVector CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_pVector, dBufferSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_pAccumulateStage1, dBufferSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_pAccumulateStage2, sizeof(uint32_t));
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
    cudaStatus = cudaMemset(dev_pAccumulateStage1, 0, dBufferSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemset(dev_pAccumulateStage2, 0, sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    reduce0<uint32_t> <<<dSliceCount, arg_dSliceSize, smemSize>>> (dev_pVector, dev_pAccumulateStage1, arg_dSize);
    reduce0<uint32_t> <<<1, dSliceCount, smemSize >>> (dev_pAccumulateStage1, dev_pAccumulateStage2, dSliceCount);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(arg_pAccumulate, dev_pAccumulateStage2, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_pVector);
    cudaFree(dev_pAccumulateStage1);
    cudaFree(dev_pAccumulateStage2);

    return cudaStatus;
}

cudaError_t CudaSparseToPackedCell(uint32_t* arg_pArrayToModify, uint32_t arg_dSize, const uint32_t arg_dSliceSize)
{
    assert(arg_dSliceSize <= 1024);

    uint32_t* dev_a = 0;
    cudaError_t cudaStatus;

    uint32_t dSliceCount = (arg_dSize + arg_dSliceSize - 1) / arg_dSliceSize;
    uint32_t dBufferSize = dSliceCount * arg_dSliceSize;


    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have arg_pVector CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for array
    cudaStatus = cudaMalloc((void**)&dev_a, dBufferSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemset(dev_a, 0, dBufferSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
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
    masked_array_to_list_first<uint32_t>
        <<<dSliceCount, arg_dSliceSize, smemSize>>>
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

cudaError_t CudaMergeCells(uint32_t* arg_pArrayToModify, uint32_t* arg_pCellLengthArray, uint32_t arg_dSize, const uint32_t arg_dCellSize, const uint32_t arg_dCellCount)
{
    cudaError_t cudaStatus = cudaError::cudaSuccess;

    uint32_t* dev_pVector = nullptr;
    uint32_t* dev_pLengthArray = nullptr;

    uint32_t dBufferSize = arg_dCellCount * arg_dCellSize;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have arg_pVector CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_pVector, dBufferSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_pLengthArray, arg_dCellCount * sizeof(uint32_t));
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
    cudaStatus = cudaMemcpy(dev_pVector, arg_pArrayToModify, arg_dSize * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_pLengthArray, arg_pCellLengthArray, arg_dCellCount * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    masked_array_to_list_second<uint32_t> <<<1, arg_dCellCount>>> (dev_pVector, dev_pLengthArray, dBufferSize, arg_dCellCount, arg_dCellSize);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(arg_pArrayToModify, dev_pVector, arg_dSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(arg_pCellLengthArray, dev_pLengthArray, arg_dCellSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_pVector);
    cudaFree(dev_pLengthArray);

    return cudaStatus;
}

cudaError_t CudaAccumulateAndPack(uint32_t* arg_pArray, const uint32_t arg_dSize)
{
    //The current code of this function only support 2^19 long arrays
    assert(arg_dSize > (1 << 19));
    cudaError_t cudaStatus = cudaError::cudaSuccess;

    uint32_t* dev_pVector = nullptr;
    uint32_t* dev_pBinSumsForFinalPacking = nullptr;
    uint32_t* dev_pFinalReduceResult = nullptr;
    uint32_t* dev_pFirstReduceResult = nullptr;

    uint32_t dSliceSize = 32;
    uint32_t dSliceCount = 0;

    if (!arg_pArray || !arg_dSize) return cudaError::cudaErrorAssert;

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
    //uint32_t smemSize = ((dSliceSize / 32) + 1) * sizeof(uint32_t);
    uint32_t dBufferSize = dSliceCount * dSliceSize;

    uint32_t dCollectStageGridSize = 1;// arg_dSize / (1024 * dSliceSize);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have arg_pVector CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_pVector, dBufferSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_pFirstReduceResult, dBufferSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_pFinalReduceResult, dSliceCount * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_pBinSumsForFinalPacking, dSliceCount * sizeof(uint32_t));
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
    cudaStatus = cudaMemcpy(dev_pVector, arg_pArray, arg_dSize * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemset(dev_pFirstReduceResult, 0, dBufferSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cg_reduce<uint32_t> <<<dSliceCount, dSliceSize, smemSize>>> (dev_pVector, dev_pFirstReduceResult, dBufferSize);
    uint32_t smemSizeForSecondStage = dSliceCount * sizeof(uint32_t);
    cg_reduce<uint32_t> <<<1, dSliceCount, smemSizeForSecondStage>>> (dev_pFirstReduceResult, dev_pFinalReduceResult, dSliceCount);
    uint32_t smemSizeForFirstStage = dSliceSize * sizeof(uint32_t);
    masked_array_to_list_first<uint32_t> <<<dSliceCount, dSliceSize, smemSizeForFirstStage >>> (dev_pVector, 0, dBufferSize);
    smemSizeForSecondStage = dSliceCount * sizeof(uint32_t);
    masked_array_to_list_second<uint32_t> <<<1, dSliceCount, smemSizeForSecondStage>>>(dev_pVector, dev_pFirstReduceResult, dBufferSize, dSliceCount, dSliceSize);

Error:
    cudaFree(dev_pVector);
    cudaFree(dev_pBinSumsForFinalPacking);
    cudaFree(dev_pFirstReduceResult);
    cudaFree(dev_pFinalReduceResult);

    return cudaStatus;

}