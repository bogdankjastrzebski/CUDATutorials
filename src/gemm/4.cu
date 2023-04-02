// 4.cu
#include <cstdio>
#include <cstdlib>
#define CHUNKSIZE 32 

__global__ void sgemm_naive(int M, int N, int K,  
							float alpha,
							const float *A, 
							const float *B,
							float beta,
							float *C) {
	
	// A and B are written row-wise.


	const uint innerRow = threadIdx.x % CHUNKSIZE;
	const uint innerCol = threadIdx.x / CHUNKSIZE; 
	
	const int numBlockSteps = (K / CHUNKSIZE) + ((K % CHUNKSIZE) > 0); 
	
	const uint cRow = blockIdx.x;
	const uint cCol = blockIdx.y; 

	const uint x = cRow * CHUNKSIZE + innerRow; 
	const uint y = cCol * CHUNKSIZE + innerCol;
		

	if (x < M && y < N) {
		
		// Jump to starting position.	
		A += cRow * CHUNKSIZE * K; // cRow * CHUNKSIZE is actual row, * K because A is row-wise; 
		B += cCol * CHUNKSIZE; // Jump to a correct column. 
		C += cRow * CHUNKSIZE * N + cCol * CHUNKSIZE; // cRow * CHUNKSIZE * N moves to correct row (N is number of columns in C), cCol * CHUNKSIZE moves to correct column 
		
		__shared__ float As[CHUNKSIZE * CHUNKSIZE];
		__shared__ float Bs[CHUNKSIZE * CHUNKSIZE];	

		
		float tmp = 0.0; // the value 
		for (int outer = 0; outer < numBlockSteps; ++outer) { 
				
			// Here we want coalescing, each thread copies one value. 	
			As[innerCol * CHUNKSIZE + innerRow]	= A[innerCol * K + innerRow];
			Bs[innerCol * CHUNKSIZE + innerRow]	= B[innerCol * N + innerRow];
			
			// Sync after copying			
			__syncthreads(); 
			
			A += CHUNKSIZE; 
			B += CHUNKSIZE * N; 	
			
			for (int inner = 0; inner < CHUNKSIZE; ++inner) {
				tmp += As[innerCol * CHUNKSIZE + inner] 
					 * Bs[inner * CHUNKSIZE + innerRow];
			}
			
			__syncthreads(); 
		}

		C[innerCol * N + innerRow] = alpha * tmp + beta * C[innerCol * N + innerRow];
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

float min(float *array, int size) {
	float ret = array[0]; 
	for (int i = 1; i < size; i++) {
		ret = ret < array[i] ? ret : array[i];
	}
	return ret; 
}

float max(float *array, int size) {
	float ret = array[0]; 
	for (int i = 1; i < size; i++) {
		ret = ret > array[i] ? ret : array[i];
	}
	return ret; 
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
	dim3 blockDim(32*32);	
	
	sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);	
	// cudaMemcpy(A, d_A, sizeof(float) * M * K, cudaMemcpyDeviceToHost); 	
	// cudaMemcpy(B, d_B, sizeof(float) * K * N, cudaMemcpyDeviceToHost); 	
	cudaMemcpy(C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);	

	printf("C[0]: %0.4f\n", C[0]);
	
	if (check(C, M * N, 4096.4096f, 0.001f)) {
		printf("Ok. \n");
	} else {
		printf("Not ok. \n");
		printf("min: %f  max: %f", min(C, M*N), max(C, M*N)); 
	} 
	
	printf("Done.\n");	

}




