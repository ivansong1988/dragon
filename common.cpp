#include "common.hpp"

#include <time.h>




int64_t Dragon::cluster_seedgen(){
    int64_t seed, pid, t;
    pid = getpid();
    t = time(0);
    seed = abs((t * 181) * ((pid - 83) * 359)) % 104729;
    return seed; 
}

rng_t* Dragon::get_rng(){
    if(!Get().random_generator){
        Get().random_generator.reset(new RNG()); 
        //reset() is the member function of share ptr object, using "." not "->"
    }
    rng_t* rng = Get().random_generator.get()->get_rng(); //get () is the member function of share ptr object
    return rng;
}

unsigned int Dragon::get_random_value(){
    rng_t* rng = get_rng();
    return (*rng)();
}

#ifdef CPU_ONLY
Dragon::Dragon() : mode(Dragon::CPU), solver_count(1), root_solver(true), cublas_handle(NULL), curand_handle(NULL){
    if(cublasCreate_v2(&cublas_handle) != CUBLAS_STATUS_SUCESS)
       LOG(ERROR) << "Coudn't create cublas handle.";
    if(curandCreateGenerator(&curand_generator, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCESS
      || curandSetPseudoRandomGeneratorSeed(curand_generator, cluster_seedgen()))
       LOG(ERROR) << "Coudn't create curand generate.";
}
Dragon::~Dragon() {
    if (cublas_handle) cublasDestroy_v2(cublas_handle);
    if (curand_generator) curandDestroyGenerator(curand_generator);
}
void Dragon::set_device(const int device_id){}
#else
Dragon::Dragon() : mode(Dragon::CPU), solver_count(1), root_solver(true){}
Dragon::~Dragon() {}
void Dragon::set_device(const int device_id){}
#endif