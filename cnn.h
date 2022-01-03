#pragma once
#ifndef __CNN__
#define __CNN__
#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

#include <CL/cl.h>
#include <stdlib.h>

//void cnn_seq(float* images, float* network, int* labels, float* confidences, int num_of_image);
void cnn(float* images, float* network, int* labels, float* confidences, int num_of_image);
void compare(const char* filename, int num_of_image);

#endif