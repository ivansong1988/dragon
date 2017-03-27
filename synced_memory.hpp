#ifndef __SYNCED_MEMORY_HPP__
#define __SYNCED_MEMORY_HPP__

#define NULL 0

inline void dragonMalloc(void **ptr, size_t size){
    *ptr = malloc(size);
    CHECK(*ptr) << "host allocation of size "<<size<<" failed";
}

inline void dragonFree(void *ptr){
    free(ptr);
}

inline void dragonMemset(void *ptr, size_t size){
    memset(ptr, 0, size);
}

inline void dragonMemcpy(void *dst, void *src, size_t size){
    memcpy(dst, src, size);
}

#ifndef CPU_ONLY
#include "cuda.h"
inline void cudaSetDevice(){
    int device;
    cudaGetDevice(&device);
    if (device != -1) return;
    CUDA_CHECK(cudaSetDevice(0));
}

inline void dragonGpuMalloc(void **ptr, size_t size){
    cudaSetDevice();
    CUDA_CHECK(cudaMalloc(ptr, size));
}

inline void dragonGpuFree(void *ptr){
    cudaSetDevice();
    CUDA_CHECK(cudaFree(ptr));
}

inline void dragonGpuMemset(void *ptr, size_t size){
    cudaSetDevice();
    CUDA_CHECK(cudaMemeset(ptr, 0, size));
}

inline void dragonGpuMemcpy(void *dst, void *src, size_t size){
    cudaSetDevice();
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
}
#endif //#ifndef CPU_ONLY

class SyncedMemory{
public: 
    SyncedMemory() : cpu_ptr(NULL), gpu_ptr(NULL), size_(0), head_(UNINITIALIZED) {}
    SyncedMemory(size_t size) : cpu_ptr(NULL), gpu_ptr(NULL), size_(size), head_(UNINITIALIZED) {}
    void to_cpu(); //state transform function
    void to_gpu(); //state transform function
    const void* cpu_data();  //const visit function
    const void* gup_data();  //const visit function
    void set_cpu_data(void *data); //share data function
    void set_gpu_data(void *data); //share data function
    void* mutable_cpu_data();  //modification function
    void* mutable_gpu_data();  //modification function
#ifndef CPU_ONLY
    void async_gpu_data(const cudaStream_t& stream);
#endif
    size_t size(){return size_;}
    SyncedHead head() {return head_;}
    ~SyncedMemory();
public:
    enum SyncedHead {UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED};
    void *cpu_ptr;
    void  *gpu_ptr;
    size_t size_;
    bool own_cpu_data, own_gpu_data;
    SyncedHead head_;
};


#endif __SYNCED_MEMORY_HPP__