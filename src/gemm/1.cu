// naive.cu / 1.cu
#include <cstdio>
#include <cstdlib>

__global__ void sgemm_naive(int M, int N, int K,  
							float alpha,
							const float *A, 
							const float *B,
							float beta,
							float *C) {

	const uint x = blockIdx.x * blockDim.x + threadIdx.x; 
	const uint y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (x < M && y < N) {
		float tmp = 0.0; 
		for (int i = 0; i < K; ++i) {
			tmp += A[x * K + i] * B[i * N + y];
		}
		C[x * N + y] = alpha * tmp + beta * C[x * N + y];
	}
}

void fill(float* M, int size, float value) {
	for (int i = 0; i < size; i++) {
		M[i] = value; 
	}
}

int CEIL_DIV(int a, int b) {
	int c = a / b; 
	int d = a % b; 
	return c + (d > 0);
}

bool check(float *array, int size, float value, float eps) {
	for (int idx = 0; idx < size; idx++) {
		if (abs(array[idx] - value) > eps) {
			return false;
		}
	}
	return true; 
}

int main() {
	const int M = 4096,
			  N = 4096,
	          K = 4096;	
	
	float *A, *d_A,
		  *B, *d_B,
          *C, *d_C;

	float alpha = 1.0f,
		  beta  = 1.0f;
				
	A = (float*)malloc(sizeof(float) * M * K);
	B = (float*)malloc(sizeof(float) * K * N);
	C = (float*)malloc(sizeof(float) * M * N);
		
	fill(A, M * K, 1.0f); 
	fill(B, K * N, 1.0f); 
	fill(C, M * N, 0.4096f); 

	cudaMalloc((void**)&d_A, sizeof(float) * M * K); 
	cudaMalloc((void**)&d_B, sizeof(float) * K * N);	
	cudaMalloc((void**)&d_C, sizeof(float) * M * N); 

	cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice); 	
	cudaMemcpy(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice); 	
	cudaMemcpy(d_C, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);	
		
	dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
	dim3 blockDim(32, 32, 1);	
	
	sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);	
	// cudaMemcpy(A, d_A, sizeof(float) * M * K, cudaMemcpyDeviceToHost); 	
	// cudaMemcpy(B, d_B, sizeof(float) * K * N, cudaMemcpyDeviceToHost); 	
	cudaMemcpy(C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);	

	printf("C[0]: %0.4f\n", C[0]);
	
	if (check(C, M * N, 4096.4096f, 0.0001f)) {
		printf("Ok. \n");
	} else {
		printf("Not ok. \n");
	} 

	printf("Done.\n");	

}




