//
// Created by 王子路 on 2022/5/13.
//

#include <iostream>
#include "Sky.cuh"
#include "src/utiles/utils.cuh"
#include "helper_cuda.h"
using namespace std;
using namespace cv;

__host__ Sky::Sky(Mat texture, float radius) {
  this->radius = radius;
  radiusSqr = radius * radius;
  if (!texture.empty()) {
	textureWidth = texture.cols;
	textureHeight = texture.rows;
	textureBitmap = getNativeTextureBitmap(texture);
  }
}

__host__ void Sky::SetAttributes(int gpuId) {
  cudaSetDevice(gpuId);
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
//  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

  const int spitch = textureBitmap.channels() * textureWidth * sizeof(unsigned char);
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModePoint;
//  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;


  checkCudaErrors(cudaMallocArray(&array_texture, &channelDesc, textureWidth, textureHeight));

  checkCudaErrors(cudaMemcpy2DToArray(array_texture,
									  0,
									  0,
									  textureBitmap.ptr(),
									  spitch,
									  textureBitmap.channels() * textureWidth * sizeof(unsigned char),
									  textureHeight,
									  cudaMemcpyHostToDevice));
  resDesc.res.array.array = array_texture;

  checkCudaErrors(cudaCreateTextureObject(&text_obj, &resDesc, &texDesc, NULL));
}

__host__ void Sky::ReleaseAttributes(int gpuId) {
  cudaSetDevice(gpuId);
  checkCudaErrors(cudaFreeArray(array_texture));
  checkCudaErrors(cudaDestroyTextureObject(text_obj));
}

__device__ ArgbColor Sky::GetColor(int side, float r, float theta, float phi, int x, int y) {
  return ArgbColor(0xFF, 0xFF, 0xFF, 0xFF);
}

__host__ __device__ Sky *Sky::SetTextureOffset(float offset) {
  textureOffset = offset;
  return this;
}

__device__ bool Sky::Hit(float3 &point, float sqrNorm, float3 &prevPoint, float prevSqrNorm,
						 float3 &velocity, SchwarzschildBlackHoleEquation *equation, float r, float theta,
						 float phi, ArgbColor &color, bool &stop, bool debug, int x, int y) {
  // Has the ray escaped to infinity?
  if (sqrNorm > radiusSqr) {
	int xPos, yPos;
	SphericalMap(textureWidth, textureHeight, r, theta, phi, xPos, yPos);
	color = AddColor(ArgbColor::fromArgb(tex2D<uchar4>(text_obj, xPos, yPos)), color);
	stop = true;
	return true;
  }
  return false;
}

//Vector3D Sky::IntersectionSearch(Vector3D prevPoint, Vector3D velocity, SchwarzschildBlackHoleEquation *equation) {
//    float stepLow = 0., stepHigh = equation.StepSize;
//    Vector3D newPoint = prevPoint;
//    Vector3D tempVelocity;
//    while (true) {
//        float stepMid = (stepLow + stepHigh) / 2.;
//        newPoint = prevPoint;
//        tempVelocity = velocity;
//        equation->Function(newPoint, tempVelocity, stepMid);
//
//        double distance = newPoint.norm2();
//        if (abs(stepHigh - stepLow) < 0.00001) {
//            break;
//        }
//        if (distance > radius) {
//            stepHigh = stepMid;
//        }
//        else {
//            stepLow = stepMid;
//        }
//    }
//    return newPoint;
//}