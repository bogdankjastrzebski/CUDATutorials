#include <cstdio> 
#define N 10000000

const int B = 256; // N / 256 + 1; 

__global__ void vector_add(float *out, float *a, float *b, int n) {
   	//int index = threadIdx.x; 
	//int stride = blockDim.x;  
	//for(int i = index; i < n; i += stride){
    //    out[i] = a[i] + b[i];
    //}

	int tid = blockDim.x * blockIdx.x + threadIdx.x; 
	for(int i = tid; i < n; i += blockDim.x * gridDim.x) {
		out[i] = a[i] + b[i];
	}
}


int main() {
	
    float *a, *b, *out; 
	float *d_a, *d_b, *d_out;

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }

	cudaMalloc((void**)&d_a, sizeof(float) * N);
	cudaMalloc((void**)&d_b, sizeof(float) * N);
	cudaMalloc((void**)&d_out, sizeof(float)*N);

	
	cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_out, out, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Main function
    vector_add<<<B,256>>>(d_out, d_a, d_b, N);

	cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

	printf("output: %f\n", out[0]);

	cudaFree(d_a);
	cudaFree(d_b); 
	cudaFree(d_out);
	
	free(a);
	free(b);
	free(out);

	printf("Done.\n");
}

