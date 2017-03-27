#ifndef __COMMON_HPP__
#define __COMMON_HPP__

class Dragon{
public:
    Dragon();
    ~Dragon();
    static Dragon& Get();
    enum Mode {CPU, GPU};
    static Mode get_mode() {return Get().mode;}
    static void set_mode(Mode mode) {Get().mode == mode;}
    static int get_solver_count() {return Get().solver_count;}
    static void set_solver_count(int count) {Get().solver_count = count;}
    static bool get_root_solver() {Get().root_solver;}
    static void set_root_solver(bool val) {Get().root_solver = val;}
    static void set_random_seed(unsigned int seed);
    static void set_device(const int device_id);
    static int64_t cluster_seedgen();
    static unsigned int get_random_value();
    static rng_t* get_rng();
#ifndef CPU_ONLY
    static cublasHandle_t get_cublas_handle() {return Get().cublas_handle;}
    static curandGenerator_t get_curand_generator(){return Get().curand_generator;}
#endif
    class RNG{
    public:
        RGN() {generator.reset(new Generator());}
        RNG(unsigned int seed) {generator.reset(new Generator(seed));}
        rng_t * get_rng(){return generator->get_rng();}
        class Generator{
        public:
            Generator():rng(new rng_t((uint32_t)Dragon::cluster_seedgen())){}
            Generator(unsigned int seed):rng(new rng_t(seed)){}
            rng_t* get_rng() {return rng.get();}
        private:
            boost::shared_ptr<rng_t> rng;
        };
    
    private:
        boost::shared_ptr<Generator> generator; //smart ptr object (not pointer)
    };
private:
    Mode mode;
    int solver_count;
    bool root_solver;
    boost::shared_ptr<RNG> random_generator;
#ifndef CPU_ONLY
    cublasHandle_t cublas_handle;
    curandGenerator_t curand_generator;
#endif
};

using std::vector;

#define INSTANTIATE_CLASS(classname) \
    template class classname<float>; \
    template class classname<double>

#define INSTANTIATE_LAYER_GPU_FORWARD(classname) \
    template void classname<float>::forward_gpu( \
        const vector<Blob<float>*>&bottom, \
        const vector<Blob<float>*>&top); \
     template void classname<double>::forward_gpu( \
        const vector<Blob<double>*>&bottom, \
        const vector<Blob<double>*>&top)

#define INSTANTIATE_LAYER_GPU_BACKWARD(classname) \
    template void classname<float>::backward_gpu( \
        const vector<Blob<float>*>&bottom, \
        const vector<bool> &data_need_bp, \
        const vector<Blob<float>*>&top); \
     template void classname<double>::backward_gpu( \
        const vector<Blob<double>*>&bottom, \
        const vector<bool> &data_need_bp, \        
        const vector<Blob<double>*>&top)

#define INSTANTIATE_LAYER_GPU_FUNCS(classname) \
    INSTANTIATE_LAYER_GPU_FORWARD(classname); \
    INSTANTIATE_LAYER_GPU_BACKWARD(classname)


#endif //__COMMON_HPP__