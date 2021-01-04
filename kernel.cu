///usr/local/cuda/bin/nvcc
template <typename T>
__global__ void addKernel(T *gpu_a, T *gpu_b, T *gpu_c, size_t s) {
        size_t index = blockIdx.x * 1024 + threadIdx.y * 32 + threadIdx.x;
        if(index < s)
                gpu_c[index] = gpu_a[index] + gpu_b[index];
}

template <typename T>
__global__ void dotKernel(T *gpu_a, T *gpu_b, T *gpu_c, size_t same_for_both, size_t length, size_t width) {
        size_t index  = blockIdx.x * 1024 + threadIdx.y * 32 + threadIdx.x;
        if(index >= length * width)
                return ;
        int a           = (int) (index / width);
        size_t index_a  = same_for_both * a;//on doit parcourir dans la longueur donc on fait juste +1
        size_t max      = index_a + same_for_both;
        size_t index_b  = (index % width);//on doit parcourir la hauteur donc on fait + width

        for(; index_a < max; ++index_a) {
                gpu_c[index] += gpu_a[index_a] * gpu_b[index_b];
                index_b += width;
        }
}

template <typename T>
__global__ void transpoKernel(T *gpu_a, T *res, size_t length, size_t width) {
        size_t index = blockIdx.x * 1024 + threadIdx.y * 32 + threadIdx.x;

        size_t i = index / width;
        size_t j = index % length;
        res[(i * length) + j] = gpu_a[(j * width) + i]  ;
}

namespace GPU {
        template <typename T>
        void add(T *a, T *b, T *res, size_t size) {
                T *gpu_a;
                T *gpu_b;
                T *gpu_c;
                cudaMalloc(&gpu_a, size * sizeof(T));
                cudaMalloc(&gpu_b, size * sizeof(T));
                cudaMalloc(&gpu_c, size * sizeof(T));

                cudaMemcpy(gpu_a, a, size * sizeof(T), cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_b, b, size * sizeof(T), cudaMemcpyHostToDevice);

                size_t nbBlocks = size / 1024 + (size % 1024 ? 1 : 0);
                dim3 grid(nbBlocks), block(32, 32);
                addKernel<<<grid, block>>>(gpu_a, gpu_b, gpu_c, size);
                cudaDeviceSynchronize();

                cudaMemcpy(res, gpu_c, size * sizeof(T), cudaMemcpyDeviceToHost);

                cudaFree(gpu_a);
                cudaFree(gpu_b);
                cudaFree(gpu_c);
        }

        template <typename T>
        void dot(T *a, T *b, T *res, size_t same_for_both, size_t length, size_t width) {
                T *gpu_a;
                T *gpu_b;
                T *gpu_c;
                cudaMalloc(&gpu_a, same_for_both * length * sizeof(T));
                cudaMalloc(&gpu_b, same_for_both * width  * sizeof(T));
                cudaMalloc(&gpu_c, length * width * sizeof(T));

                cudaMemcpy(gpu_a, a, same_for_both * length * sizeof(T), cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_b, b, same_for_both * width  * sizeof(T), cudaMemcpyHostToDevice);

                size_t nbBlocks = length * width / 1024 + (length * width % 1024 ? 1 : 0);
                dim3 grid(nbBlocks), block(32, 32);
                dotKernel<<<grid, block>>>(gpu_a, gpu_b, gpu_c, same_for_both, length, width);
                cudaDeviceSynchronize();

                cudaMemcpy(res, gpu_c, length * width * sizeof(T), cudaMemcpyDeviceToHost);

                cudaFree(gpu_a);
                cudaFree(gpu_b);
                cudaFree(gpu_c);
        }

        template <typename T>
        void transpose(T *a, size_t length, size_t width) {
                T *gpu_a;
                T *res;
                cudaMalloc(&gpu_a, length * width * sizeof(T));
                cudaMalloc(&res, length * width * sizeof(T));
                cudaMemcpy(gpu_a, a, length * width * sizeof(T), cudaMemcpyHostToDevice);

                size_t nbBlocks = length * width / 1024 + (length * width % 1024 ? 1 : 0);
                dim3 grid(nbBlocks), block(32, 32);
                transpoKernel<<<grid, block>>>(gpu_a, res, length, width);

                cudaMemcpy(a, res, length * width * sizeof(T), cudaMemcpyDeviceToHost);
                cudaFree(gpu_a);
                cudaFree(res);
        }
};

template void GPU::add(int *, int *, int *, size_t);
template void GPU::add(double *, double *, double *, size_t);
template void GPU::add(short *, short *, short *, size_t);
template void GPU::add(char *, char *, char *, size_t);
template void GPU::add(long *, long *, long *, size_t);
template void GPU::add(float *, float *, float *, size_t);

template void GPU::dot(int *, int *, int *, size_t, size_t, size_t);
template void GPU::dot(float *, float *, float *, size_t, size_t, size_t);
template void GPU::dot(char *, char *, char *, size_t, size_t, size_t);
template void GPU::dot(short *, short *, short *, size_t, size_t, size_t);
template void GPU::dot(double *, double *, double *, size_t, size_t, size_t);
template void GPU::dot(long *, long *, long *, size_t, size_t, size_t);

template void GPU::transpose(int *, size_t, size_t);
