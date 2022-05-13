//
// Created by 王子路 on 2022/5/13.
//

#ifndef BLACKHOLERAYTRACER_TEXTUREDDISK_H
#define BLACKHOLERAYTRACER_TEXTUREDDISK_H
#include "IHitable.cuh"
#include "SchwarzschildBlackHoleEquation.cuh"
#include <iostream>

//using namespace Models;
using namespace std;
using namespace cv;

class TexturedDisk : public IHitable {
 private:
  int textureWidth;
  int textureHeight;
  cudaArray_t array_texture_disk;
  Mat textureBitmap;
  cudaTextureObject_t tex_obj = 0;

  float radiusInner;
  float radiusOuter;
  float radiusInnerSqr;
  float radiusOuterSqr;

 public:
  __host__ TexturedDisk(float radiusInner, float radiusOuter, Mat texture);
  __host__ void SetAttributes(int gpuId);
  __host__ void ReleaseAtrributes(int gpuId);

  __device__ bool Hit(float3 &point, float sqrNorm, float3 &prevPoint, float prevSqrNorm,
					  float3 &velocity, SchwarzschildBlackHoleEquation *equation, float r, float theta,
					  float phi, ArgbColor &color, bool &stop, bool debug, int x, int y);

 protected:
  __host__ __device__ float3 IntersectionSearch(int side,
												float3 prevPoint,
												float3 velocity,
												SchwarzschildBlackHoleEquation *equation);
 public:
  __device__ ArgbColor GetColor(int side, float r, float theta, float phi, int x, int y);

};

#endif //BLACKHOLERAYTRACER_TEXTUREDDISK_H
