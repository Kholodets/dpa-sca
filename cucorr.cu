#include <R.h>

/*
*  H = matrix of hypothetical power values
*  T = matrix of measured power traces
*  R = matrix of correlations
*  n_d = number of power traces/hyp values
*  n_t = total time of each power trace/how many samples
*/
//I think this is a pretty naive implementation because each thread just solves the correlation sequentially
//could possibly be optimized by
//a. caching repeatedly used summations
//b. parallelize the summation: schedule kernels to run prefix sums
__global__ void corr(int *H, double *T, int n_d, int n_t, double *R)
{
	long i = threadIdx.x; //key
	long j = blockIdx.x; //the sample time
	//get the averages

	double h_bar = 0;
	double t_bar = 0;
	for(long d = 0; d < n_d; d++) {
		h_bar += H[256 * d + i];
		t_bar += T[(long)n_t * d + j];
	}
	h_bar = h_bar / n_d;
	t_bar = t_bar / n_d;


	double num = 0;
	double den_h = 0;
	double den_t = 0;

	for(long d = 0; d < n_d; d++) {
		double hc = ((double) H[256 * d + i]) - h_bar;
		double tc = (T[(long) n_t * d + j]) - t_bar;
		num += hc * tc;
		den_h += hc * hc;
		den_t += tc * tc;
	}

	double den = sqrt(den_h * den_t);
	

	R[(long)n_t * i + j] = num/den;
}

extern "C"
void pcorr(int *h_H, double *h_T, int *h_n_d, int *h_n_t, double *h_R)
{
	size_t Hsize = sizeof(int) * *h_n_d * 256;
	size_t Tsize = sizeof(double) * *h_n_d * *h_n_t;
	size_t Rsize = sizeof(double) * 256 * *h_n_t;

	int *d_H;
	double *d_T, *d_R;

	cudaMalloc(&d_H, Hsize);
	cudaMalloc(&d_T, Tsize);
	cudaMalloc(&d_R, Rsize);

	cudaMemcpy(d_H, h_H, Hsize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_T, h_T, Tsize, cudaMemcpyHostToDevice);

	int tpb = 256;
	int bpg = *h_n_t;
	corr<<<bpg,tpb>>>(d_H, d_T, *h_n_d, *h_n_t, d_R);
	cudaError e = cudaGetLastError();
	if(e) {
		printf("pcorr encountered a cuda error:\n");
		printf(cudaGetErrorString(e));
	}

	cudaMemcpy(h_R, d_R, Rsize, cudaMemcpyDeviceToHost);


	cudaFree(d_H);
	cudaFree(d_T);
	cudaFree(d_R);
}


