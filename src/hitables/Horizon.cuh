//
// Created by 王子路 on 2022/5/13.
//

#ifndef HORIZON_H
#define HORIZON_H

#include <iostream>
#include "IHitable.cuh"
#include "src/utiles/ArgbColor.cuh"

//using namespace Models;
using namespace std;
using namespace cv;

class Horizon : public IHitable {

 private:
  bool checkered;
  int textureWidth;
  int textureHeight;
  Mat textureBitmap;
  cudaTextureObject_t tex_obj;
  cudaArray_t array_texture;

  double radius = 1.0;
  bool textureBitmapIsNull = true;

 public:
  __host__ Horizon(Mat texture, bool checkered);
  __host__ void SetAttributes(int gpuId);
  __host__ void ReleaseAttributes(int gpuId);

  __device__ bool Hit(float3 &point, float sqrNorm, float3 &prevPoint, float prevSqrNorm,
							   float3 &velocity, SchwarzschildBlackHoleEquation *equation, float r, float theta,
							   float phi, ArgbColor &color, bool &stop, bool debug, int x, int y);
 public:
  __device__  ArgbColor GetColor(int side, float r, float theta, float phi, int x, int y);

 protected:
  __host__ __device__ float3 IntersectionSearch(float3 prevPoint, float3 velocity, SchwarzschildBlackHoleEquation *equation);

};

#endif /* HORIZON_H */