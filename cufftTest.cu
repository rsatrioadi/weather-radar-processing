#include <cufft.h>

#include <iostream>
#include <complex>

// #define DATA_LEN 1024
// #define ITERATION 100000

int main(int argc, char **argv)
{
	if (argc != 3)
	{
		std::cout << "Usage: " << argv[0] << " [DATA_LEN] [ITERATION]" << std::endl;
		return 1;
	}
	int DATA_LEN = atoi(argv[1]);
	int ITERATION = atoi(argv[2]);
	// Siapkan memory untuk input data di Host
	// cufftComplex *t_HostInputData = (cufftComplex*)malloc(sizeof(cufftComplex)*DATA_LEN*1);
	cufftComplex *t_HostInputData;
	cudaError_t status = cudaMallocHost((void**)&t_HostInputData, sizeof(cufftComplex)*DATA_LEN*1);
	for(int i=0; i < DATA_LEN; i++)
	{
		t_HostInputData[i].x = 1.0;
		t_HostInputData[i].y = 1.0;
	}

	// for(int i=0; i < DATA_LEN; i++)
	// {
	// 	std::cout << t_HostInputData[i].x << " + i" << t_HostInputData[i].y << std::endl;
	// }
	// std::cout << std::endl;

	// Siapkan memory untuk data di GPU
	cufftComplex *t_InputData;
	cufftComplex *t_OutputData;
	cudaMalloc((void**)&t_InputData, sizeof(cufftComplex)*DATA_LEN*1);
	cudaMalloc((void**)&t_OutputData, sizeof(cufftComplex)*DATA_LEN*1);
	if (cudaGetLastError() != cudaSuccess)
	{
		std::cout << "Cuda error: Failed to allocate" << std::endl;
		return 1;
	}
	cudaMemset(t_InputData, 0, DATA_LEN);
	cudaMemcpy(t_InputData, t_HostInputData, sizeof(cufftComplex)*DATA_LEN*1, cudaMemcpyHostToDevice);
	cudaMemset(t_OutputData, 0, DATA_LEN);

	// FFT plan
	cufftHandle t_Plan;
	if (cufftPlan1d(&t_Plan, DATA_LEN, CUFFT_C2C, 1) != CUFFT_SUCCESS)
	{
			std::cout << "CUFFT error: Plan creation failed" << std::endl;
			return 1;
	}

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float elapsedTime;
	cudaEventRecord(start, 0);

	// Execute FFT Forward operation
	for(int i=0; i < ITERATION; i++)
	{
		if (cufftExecC2C(t_Plan, t_InputData, t_OutputData, CUFFT_FORWARD) != CUFFT_SUCCESS)
		{
			std::cout << "CUFFT error: ExecC2C Forward failed" << std::endl;
			return 1;
		}
	}

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&elapsedTime, start, end);
	printf("%d times for the FFT: %fms\n", ITERATION, elapsedTime);

	// // Execute FFT Backward / IFFT operation
	// if (cufftExecC2C(t_Plan, t_OutputData, t_InputData, CUFFT_INVERSE) != CUFFT_SUCCESS)
	// {
	// 	std::cout << "CUFFT error: ExecC2C Forward failed" << std::endl;
	// 	return 1;
	// }

	// Synchro
	if (cudaDeviceSynchronize() != cudaSuccess)
	{
		std::cout << "Cuda error: Failed to synchronize" << std::endl;
		return 1;
	}

	// Siapkan host memory untuk menerima result FFT dari GPU
	cufftComplex *t_HostData = (cufftComplex*)malloc(sizeof(cufftComplex)*DATA_LEN*1);

	// Copy from GPU to host memroy
	cudaMemcpy(t_HostData, t_OutputData, sizeof(cufftComplex)*DATA_LEN*1, cudaMemcpyDeviceToHost);

	// Display data
	// for(int i=0; i < DATA_LEN; i++)
	// {
	// 	std::cout << t_HostData[i].x << " + i" << t_HostData[i].y << std::endl;
	// }

	// Cleaning stuff
	cufftDestroy(t_Plan);
	cudaFree(t_InputData);
	cudaFree(t_OutputData);
	cudaFreeHost(t_HostInputData);

	return 0;
}
