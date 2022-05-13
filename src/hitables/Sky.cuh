//
// Created by 王子路 on 2022/5/13.
//

#ifndef SKY_H
#define SKY_H

#include <opencv2/opencv.hpp>
#include <iostream>

#include "IHitable.cuh"


//using namespace Models;
using namespace std;
using namespace cv;

class Sky : public IHitable {

 private:
  int textureWidth;
  int textureHeight;
  Mat textureBitmap;
  cudaTextureObject_t text_obj = 0;
  cudaArray_t array_texture;
  double textureOffset = 0;
  double radius;
  double radiusSqr;

 public:
  __host__ Sky(Mat texture, float radius);

  __host__ void SetAttributes(int gpuId);
  __host__ void ReleaseAttributes(int gpuId);

  __host__ __device__ Sky *SetTextureOffset(float offset);

  __device__ bool Hit(float3 &point, float sqrNorm, float3 &prevPoint, float prevSqrNorm,
					  float3 &velocity, SchwarzschildBlackHoleEquation *equation, float r, float theta,
					  float phi, ArgbColor &color, bool &stop, bool debug, int x, int y);

  __device__  ArgbColor GetColor(int side, float r, float theta, float phi, int x, int y);
//protected:
//    Vector3D IntersectionSearch(Vector3D prevPoint, Vector3D velocity, SchwarzschildBlackHoleEquation equation);

};

#endif /* SKY_H */