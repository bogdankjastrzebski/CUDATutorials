#include <cstdio>

__global__ void cuda_hello() {
	printf("Hello from the other side!");
}

int main() {
	cuda_hello<<<1,1>>>();
	printf("Done.\n");
	return 0;	
}
