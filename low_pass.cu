#include <R.h>

__global__ void lp(double *T, double a, int n_d, int n_t, int tpb, double *lpT)
{
	long th = threadIdx.x;
	long bl = blockIdx.x;

	long n = tpb * bl + th;
	//printf("i ran :), n = %ld, n_t = %d\n", n, n_t);

	if (n >= n_d) {
		return;
	}

	long d = n * n_t;

	lpT[d] = T[d];

	for (int i = 1; i < n_t; i++) {
		lpT[d + i] = a * (double) lpT[d+i-1] + (1.0 - a) * T[d+i];
	}
}

extern "C"
void low_pass(double *h_T, double *h_a, int *h_n_d, int *h_n_t, double *h_lpT)
{
	size_t Tsize = sizeof(double) * *h_n_d * *h_n_t;

	double *d_T, *d_lpT;

	cudaMalloc(&d_T, Tsize);
	cudaMalloc(&d_lpT, Tsize);

	cudaMemcpy(d_T, h_T, Tsize, cudaMemcpyHostToDevice);

	int tpb = 1024;
	int bpg = 1 + *h_n_d / 1024;
	lp<<<bpg,tpb>>>(d_T, *h_a, *h_n_d, *h_n_t, tpb, d_lpT);
	cudaError e = cudaGetLastError();
	if(e) {
		printf(cudaGetErrorString(e));
	}

	cudaMemcpy(h_lpT, d_lpT, Tsize, cudaMemcpyDeviceToHost);

	cudaFree(d_T);
	cudaFree(d_lpT);
}


