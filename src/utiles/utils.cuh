#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include "vector_types.h"
#include "helper_math.h"
//#include "helper_cuda.h"
#include "ArgbColor.cuh"
__host__ __device__ void ToSpherical(const float3 v, float &r, float &theta, float &phi);
__host__ __device__ ArgbColor AddColor(ArgbColor hitColor, ArgbColor tintColor);

__host__ __device__ void ToSpherical(const float3, float &, float &, float &);
__host__ __device__ float DoubleMod(float n, float m);
__host__ __device__ void SphericalMap(int SizeX, int SizeY, float r, float theta, float phi, int &x, int &y);
__host__ __device__ void DiskMap(float rMin,
								 float rMax,
								 int SizeX,
								 int SizeY,
								 float r,
								 float theta,
								 float phi,
								 int &x,
								 int &y);
__host__ cv::Mat getNativeTextureBitmap(cv::Mat);
#endif /* UTILS_H */