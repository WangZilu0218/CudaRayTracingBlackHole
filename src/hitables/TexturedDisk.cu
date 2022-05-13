//
// Created by 王子路 on 2022/5/13.
//

#include "TexturedDisk.cuh"
#include "src/utiles/utils.cuh"
#include "helper_cuda.h"

__host__ TexturedDisk::TexturedDisk(float radiusInner, float radiusOuter, Mat texture) {
  this->radiusInner = radiusInner;
  this->radiusOuter = radiusOuter;
  radiusInnerSqr = radiusInner * radiusInner;
  radiusOuterSqr = radiusOuter * radiusOuter;
  textureWidth = texture.cols;
  textureHeight = texture.rows;
  textureBitmap = texture.clone();
}

__host__ void TexturedDisk::SetAttributes(int gpuId) {
  cudaSetDevice(gpuId);
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
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

  checkCudaErrors(cudaMallocArray(&array_texture_disk, &channelDesc, textureWidth, textureHeight));
  checkCudaErrors(cudaMemcpy2DToArray(array_texture_disk,
									  0,
									  0,
									  textureBitmap.ptr(),
									  spitch,
									  textureBitmap.channels() * textureWidth * sizeof(unsigned char),
									  textureHeight,
									  cudaMemcpyHostToDevice));
  resDesc.res.array.array = array_texture_disk;
  checkCudaErrors(cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, NULL));
}

__host__ void TexturedDisk::ReleaseAtrributes(int gpuId) {
  cudaSetDevice(gpuId);
  checkCudaErrors(cudaFreeArray(array_texture_disk));
  checkCudaErrors(cudaDestroyTextureObject(tex_obj));
}


__device__ ArgbColor TexturedDisk::GetColor(int side, float r, float theta, float phi, int x, int y) {
  int xPos, yPos;
  DiskMap(radiusInner, radiusOuter, textureWidth, textureHeight, r, theta, phi, xPos, yPos);
  return ArgbColor::fromArgb(tex2D<uchar4>(tex_obj, xPos, yPos));
  // row major order so like this apparently. needs testing
  // original code: return Color.FromArgb(textureBitmap[yPos * textureWidth + xPos]);
}

__device__ bool TexturedDisk::Hit(float3 &point, float sqrNorm, float3 &prevPoint, float prevSqrNorm,
								  float3 &velocity, SchwarzschildBlackHoleEquation *equation, float r, float theta,
								  float phi, ArgbColor &color, bool &stop, bool debug, int x, int y) {
  // Remember what side of the plane we're currently on, so that we can detect
  // whether we've crossed the plane after stepping.
  int side = prevPoint.y > 0 ? -1 : prevPoint.y < 0 ? 1 : 0;

  // Did we cross the horizontal plane?
  bool success = false;
  if (point.y * side >= 0) {
	float3 colpoint = IntersectionSearch(side, prevPoint, velocity, equation);
	float colpointsqr = dot(colpoint, colpoint);
	if ((colpointsqr >= radiusInnerSqr) && (colpointsqr <= radiusOuterSqr)) {
	  float tempR = 0;
	  float tempTheta = 0;
	  float tempPhi = 0;
	  ToSpherical(colpoint, tempR, tempTheta, tempPhi);
	  color = AddColor(GetColor(side, tempR, tempPhi, tempTheta + M_PI / 12, x, y), color);
	  stop = false;
	  success = true;
	}
  }
  return success;
}

__host__ __device__ float3 TexturedDisk::IntersectionSearch(int side,
															float3 prevPoint,
															float3 velocity,
															SchwarzschildBlackHoleEquation *equation) {
  float stepLow = 0, stepHigh = equation->StepSize;
  float3 newPoint = prevPoint;
  float3 tempVelocity;
  while (true) {
	float stepMid = (stepLow + stepHigh) / 2;
	newPoint = prevPoint;
	tempVelocity = velocity;
	equation->Function(newPoint, tempVelocity, stepMid);
	if (abs(stepHigh - stepLow) < 0.00001) {
	  break;
	}
	if (side * newPoint.y > 0) {
	  stepHigh = stepMid;
	} else {
	  stepLow = stepMid;
	}
  }
  return newPoint;
}
