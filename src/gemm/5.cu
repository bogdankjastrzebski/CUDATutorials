// 5.cu
#include <cstdio>
#include <cstdlib>
#define BN 128
#define BM 128 
#define BK 8
#define TM 8
#define TN 8

__global__ void sgemm_naive(int M, int N, int K,  
							float alpha,
							const float *A, 
							const float *B,
							float beta,
							float *C) {
	
	// A and B are written row-wise.
	
	const uint numThreadsBlocktile = BM * BN / (TM * TN);
	const uint strideA = numThreadsBlocktile / BK; 
	const uint strideB = numThreadsBlocktile / BN;	

	const uint innerColA = threadIdx.x % BK;
	const uint innerRowA = threadIdx.x / BK; 
	
	const uint innerColB = threadIdx.x % BN;
	const uint innerRowB = threadIdx.x / BN; 
	
	const uint cRow = blockIdx.y;
	const uint cCol = blockIdx.x; 
	
	const int threadCol = threadIdx.x % (BN / TN); 
	const int threadRow = threadIdx.x / (BN / TN);	

	const uint x = cRow * BM + innerRowA; 
	const uint y = cCol * BN + innerColB;	

	if (x < M && y < N) {
		
		// Jump to starting position.	
		A += cRow * BM * K; // cRow * CHUNKSIZE is actual row, * K because A is row-wise; 
		B += cCol * BN; // Jump to a correct column. 
		C += cRow * BM * N + cCol * BN; // cRow * CHUNKSIZE * N moves to correct row (N is number of columns in C), cCol * CHUNKSIZE moves to correct column 
		
		__shared__ float As[BM * BK];
		__shared__ float Bs[BK * BN];	

		float threadResults[TM * TN] = {0.0}; // Values
		float regM[TM] = {0.0};	
		float regN[TN] = {0.0};

		for (int bkIdx = 0; bkIdx < K; bkIdx += BK) { 
			// Here we want coalescing, each thread copies one value. 	
			for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
				As[(innerRowA + loadOffset) * BK + innerColA] =
					A[(innerRowA + loadOffset) * K + innerColA];
			}
			for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
				Bs[(innerRowB + loadOffset) * BN + innerColB] = 
					B[(innerRowB + loadOffset) * N + innerColB];
			}
			
			// Sync after copying			
			__syncthreads(); 
			
			A += BK; // CHUNKSiZE is BK  
			B += BK * N;	
			
			for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
				/*
				Each iteration we are doing an outer product:
				 ___   [b_1 ... b_TN]
				|a_1| 				
				  .
                  .
                  .
                |a_TM|

				*/ 	
				// load to stack	
				for (uint i = 0; i < TM; ++i) {
					regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
				}
				for (uint i = 0; i < TN; ++i) {
					regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
				}	
				
				for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {	
					for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {	
						threadResults[resIdxM * TN + resIdxN] +=
							regM[resIdxM] * regN[resIdxN];
					}
				}
			}
	
			__syncthreads(); 
		}
		
		for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {	
			for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
				C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
					alpha * threadResults[resIdxM] +
					beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
			
			}
		}
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
		
	dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN), 1);
	dim3 blockDim(BM * BN / (TM * TN));	
	
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




