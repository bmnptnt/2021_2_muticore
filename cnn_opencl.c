#pragma warning(disable : 4996)
#include "cnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <windows.h>
#include <math.h>
#include <time.h>
#include <direct.h>
#include <CL/cl.h>
extern const char* CLASS_NAME[];
#define BATCH 30
cl_int err;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
size_t kernel_source_size;
cl_program program;
cl_kernel kernel_cnv,kernel_cnvSum,kernel_pool,kernel_fc;
cl_mem buf_in, buf_filter,buf_fout,buf_sumOut,buf_bias;//convolution layer
cl_mem buf_poolin, buf_poolout;//pooling layer
cl_mem buf_fcin, buf_fcout, buf_fcB, buf_fcW;//fc layer
size_t  global[3] = { BATCH, 0, 0 }, local[3] = {1,0,0};
char* get_source_code(const char* file_name, size_t* len) {
	FILE* file = fopen(file_name, "rb");
	if (file == NULL) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}

	fseek(file, 0, SEEK_END);
	size_t length = (size_t)ftell(file);
	rewind(file);

	char* source_code = (char*)malloc(length + 1);
	fread(source_code, length, 1, file);
	source_code[length] = '\0';
	fclose(file);
	*len = length;

	return source_code;
}

void build_error(cl_program program, cl_device_id device, cl_int err) {
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		char* log;

		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		CHECK_ERROR(err);

		log = (char*)malloc(log_size + 1);
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		CHECK_ERROR(err);

		log[log_size] = '\0';
		printf("Compiler error:\n%s\n", log);
		free(log);
		exit(0);
	};
}

static void cl_cnvs(float* inputs, float* outputs, float* filter, float* biases, int inDim, int outDim, int nbyn) {
	memset(outputs, 0, BATCH*nbyn * nbyn * outDim * sizeof(float));
	int offset = nbyn * nbyn;

	
	buf_fout = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*BATCH * outDim * inDim * offset, NULL, &err);
	CHECK_ERROR(err);
	buf_sumOut = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * BATCH * outDim * offset, NULL, &err);
	CHECK_ERROR(err);
	buf_in = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * BATCH * inDim * offset, NULL, &err);
	CHECK_ERROR(err);
	buf_filter = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDim * inDim * 3 * 3, NULL, &err);
	CHECK_ERROR(err);
	buf_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDim, NULL, &err);
	CHECK_ERROR(err);


	err = clEnqueueWriteBuffer(queue, buf_in, CL_FALSE, 0, sizeof(float) * BATCH * inDim * offset, inputs, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, buf_filter, CL_TRUE, 0, sizeof(float) * inDim * outDim * 3 * 3, filter, 0, NULL, NULL);
	CHECK_ERROR(err);



	err = clSetKernelArg(kernel_cnv, 0, sizeof(cl_mem), &buf_fout);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_cnv, 1, sizeof(cl_mem), &buf_in);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_cnv, 2, sizeof(cl_mem), &buf_filter);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_cnv, 3, sizeof(int), &inDim);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_cnv, 4, sizeof(int), &outDim);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_cnv, 5, sizeof(int), &nbyn);
	CHECK_ERROR(err);

	global[1] = outDim * inDim;
	global[2] = offset;
	

	local[1] = 8;
	local[2] = 4;
	err = clEnqueueNDRangeKernel(queue, kernel_cnv, 3, NULL, global, local, 0, NULL, NULL);
	CHECK_ERROR(err);

	err = clEnqueueWriteBuffer(queue, buf_bias, CL_TRUE, 0, sizeof(float) * outDim, biases, 0, NULL, NULL);
	CHECK_ERROR(err);
	
	global[1] = outDim * inDim * offset;
	local[1] = inDim;
	err = clSetKernelArg(kernel_cnvSum, 0, sizeof(cl_mem), &buf_fout);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_cnvSum, 1, sizeof(cl_mem), &buf_sumOut);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_cnvSum, 2, sizeof(float) * local[1], NULL);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_cnvSum, 3, sizeof(int), &global[1]);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_cnvSum, 4, sizeof(cl_mem), &buf_bias);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_cnvSum, 5, sizeof(int), &offset);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_cnvSum, 6, sizeof(int), &outDim);
	CHECK_ERROR(err);
	err = clEnqueueNDRangeKernel(queue, kernel_cnvSum, 2, NULL, global, local, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueReadBuffer(queue, buf_sumOut, CL_TRUE, 0, sizeof(float) * BATCH * outDim * offset, outputs, 0, NULL, NULL);
	CHECK_ERROR(err);

	clFinish(queue);

	clReleaseMemObject(buf_fout);
	clReleaseMemObject(buf_in);
	clReleaseMemObject(buf_filter);
	clReleaseMemObject(buf_sumOut);
	clReleaseMemObject(buf_bias);
	
}

static void cl_pooling(float* input, float* output, int DIM, int nbyn) {
	int offset = nbyn * nbyn;
	buf_poolin= clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * BATCH * DIM*offset, NULL, &err);
	CHECK_ERROR(err);
	buf_poolout = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * BATCH * (DIM * offset)/4, NULL, &err);
	CHECK_ERROR(err);

	global[1] = DIM;
	global[2] = offset/4;
	local[1] = 1;
	local[2] = 1;
	err = clEnqueueWriteBuffer(queue, buf_poolin, CL_TRUE, 0, sizeof(float) * BATCH * DIM * offset, input, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_pool, 0, sizeof(cl_mem), &buf_poolin);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_pool, 1, sizeof(cl_mem), &buf_poolout);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_pool, 2, sizeof(int), &DIM);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_pool, 3, sizeof(int), &nbyn);
	CHECK_ERROR(err);
	err = clEnqueueNDRangeKernel(queue, kernel_pool, 3, NULL, global, local, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueReadBuffer(queue, buf_poolout, CL_TRUE, 0, sizeof(float) * BATCH * (DIM * offset) / 4, output, 0, NULL, NULL);
	CHECK_ERROR(err);
	clReleaseMemObject(buf_poolin);
	clReleaseMemObject(buf_poolout);
}

void cl_fc(float* input, float* output, float* weights, float* biases, int inDim, int outDim) {
	buf_fcin = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * BATCH * inDim, NULL, &err);
	CHECK_ERROR(err);
	buf_fcout = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * BATCH * outDim, NULL, &err);
	CHECK_ERROR(err);
	buf_fcW = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDim*inDim, NULL, &err);
	CHECK_ERROR(err);
	buf_fcB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDim, NULL, &err);
	CHECK_ERROR(err);
	global[1] = outDim * inDim;
	local[1] = inDim;
	err = clEnqueueWriteBuffer(queue, buf_fcin, CL_TRUE, 0, sizeof(float) * BATCH *inDim, input, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, buf_fcW, CL_TRUE, 0, sizeof(float) *outDim*inDim, weights, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, buf_fcB, CL_TRUE, 0, sizeof(float) * outDim, biases, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_fc, 0, sizeof(cl_mem), &buf_fcin);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_fc, 1, sizeof(cl_mem), &buf_fcout);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_fc, 2, sizeof(cl_mem), &buf_fcW);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_fc, 3, sizeof(cl_mem), &buf_fcB);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_fc, 4, sizeof(int), &inDim);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_fc, 5, sizeof(int), &outDim);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_fc, 6, sizeof(int)*local[1],NULL);
	CHECK_ERROR(err);
	err = clEnqueueNDRangeKernel(queue, kernel_fc, 2, NULL, global, local, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueReadBuffer(queue, buf_fcout, CL_TRUE, 0, sizeof(float) * BATCH *outDim, output, 0, NULL, NULL);
	CHECK_ERROR(err);

	clReleaseMemObject(buf_fcB);
	clReleaseMemObject(buf_fcW);
	clReleaseMemObject(buf_fcin);
	clReleaseMemObject(buf_fcout);
}

static void softmax(float* input, int N) {
	int i;
	float max = input[0];
	for (i = 1; i < N; i++) {
		if (max < input[i]) max = input[i];
	}
	float sum = 0;
	for (i = 0; i < N; i++) {
		sum += exp(input[i] - max);
	}
	for (i = 0; i < N; i++) {
		input[i] = exp(input[i] - max) / (sum + 1e-7);
	}
}

static int find_max(float* input, int classNum) {
	int i;
	int maxIndex = 0;
	float max = 0;
	for (i = 0; i < classNum; i++) {
		if (max < input[i]) {
			max = input[i];
			maxIndex = i;
		}
	}
	return maxIndex;
}

const int INPUT_DIM[] = {
	3, 64,
	64,

	64,128,
	128,

	128, 256, 256,
	256,

	256, 512, 512,
	512,

	512, 512, 512,
	512,

	512,
	512,
	512
};

const int OUTPUT_DIM[] = {
	64, 64,
	64,

	128, 128,
	128,

	256, 256, 256,
	256,

	512, 512, 512,
	512,

	512, 512, 512,
	512,

	512,
	512,
	10
};

const int NBYN[] = {
	32, 32,
	16,

	16, 16,
	8,

	8, 8, 8,
	4,

	4, 4, 4,
	2,

	2, 2, 2,
	1,

	1,
	1,
	1
};


void cnn_init() {

	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);


	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
	CHECK_ERROR(err);

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
	CHECK_ERROR(err);


	char* kernel_source = get_source_code("kernel.cl", &kernel_source_size);
	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);

	err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
	build_error(program, device, err);
	CHECK_ERROR(err);

	kernel_cnv = clCreateKernel(program, "cnvs", &err);
	CHECK_ERROR(err);
	kernel_cnvSum = clCreateKernel(program, "cnvSum", &err);
	CHECK_ERROR(err);
	kernel_pool = clCreateKernel(program, "pooling", &err);
	CHECK_ERROR(err);
	kernel_fc = clCreateKernel(program, "fc", &err);
	CHECK_ERROR(err);
}

void cnn(float* images, float** network, int* labels, float* confidences, int num_of_image) {
	float* w[21];
	float* b[21];
	int offset = 0;
	// link weights and biases to network
	for (int i = 0; i < 17; ++i) {
		if (i == 2 || i == 5 || i == 9 || i == 13) i++;	// pooling layer has no weights and biases
		w[i] = network + offset;
		offset += 3 * 3 * INPUT_DIM[i] * OUTPUT_DIM[i];
		b[i] = network + offset;
		offset += OUTPUT_DIM[i];
	}
	for (int i = 18; i < 21; ++i) {
		w[i] = network + offset;
		offset += INPUT_DIM[i] * OUTPUT_DIM[i];
		b[i] = network + offset;
		offset += OUTPUT_DIM[i];
	}


	// allocate memory for layer
	float* layer[21];
	for (int i = 0; i < 21; ++i) {
		layer[i] = (float*)malloc(sizeof(float)*BATCH * OUTPUT_DIM[i] * NBYN[i] * NBYN[i]);
		if (layer[i] == NULL) {
			perror("malloc error");
		}
	}

	time_t start, end;
	start = clock();
	// run network
	for (int i = 0; i < num_of_image/BATCH; ++i) 
	{
	
		cl_cnvs(images, layer[0], w[0], b[0], INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0]);
		cl_cnvs(layer[0], layer[1], w[1], b[1], INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1]);
		cl_pooling(layer[1], layer[2], INPUT_DIM[2], NBYN[2] * 2);
	
	
		cl_cnvs(layer[2], layer[3], w[3], b[3], INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3]);
		cl_cnvs(layer[3], layer[4], w[4], b[4], INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4]);
		cl_pooling(layer[4], layer[5], INPUT_DIM[5], NBYN[5] * 2);


		cl_cnvs(layer[5], layer[6], w[6], b[6], INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6]);
		cl_cnvs(layer[6], layer[7], w[7], b[7], INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7]);
		cl_cnvs(layer[7], layer[8], w[8], b[8], INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8]);
		cl_pooling(layer[8], layer[9], INPUT_DIM[9], NBYN[9] * 2);


		cl_cnvs(layer[9], layer[10], w[10], b[10], INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10]);
		cl_cnvs(layer[10], layer[11], w[11], b[11], INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11]);
		cl_cnvs(layer[11], layer[12], w[12], b[12], INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12]);
		cl_pooling(layer[12], layer[13], INPUT_DIM[13], NBYN[13] * 2);


		cl_cnvs(layer[13], layer[14], w[14], b[14], INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14]);
		cl_cnvs(layer[14], layer[15], w[15], b[15], INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15]);
		cl_cnvs(layer[15], layer[16], w[16], b[16], INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16]);
		cl_pooling(layer[16], layer[17], INPUT_DIM[17], NBYN[17] * 2);

		
		cl_fc(layer[17], layer[18], w[18], b[18], INPUT_DIM[18], OUTPUT_DIM[18]);
		cl_fc(layer[18], layer[19], w[19], b[19], INPUT_DIM[19], OUTPUT_DIM[19]);
		cl_fc(layer[19], layer[20], w[20], b[20], INPUT_DIM[20], OUTPUT_DIM[20]);
	


		for (int j = 0; j < BATCH; j++) 
		{
			softmax(layer[20]+j*10, 10);
			labels[i*BATCH+j] = find_max(layer[20]+j*10, 10);
			confidences[i * BATCH + j] = layer[20][labels[i * BATCH + j] + j * 10];
		}
	
		images += 32 * 32 * 3*BATCH;
	}
	end = clock();
	printf("Elapsed time: %.2f sec\n", (double)(end - start) / CLK_TCK);

	

	for (int i = 0; i < 21; ++i) 
	{
		free(layer[i]);
	}

	clReleaseKernel(kernel_cnv);
	clReleaseKernel(kernel_cnvSum);
	clReleaseKernel(kernel_fc);
	clReleaseKernel(kernel_pool);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}