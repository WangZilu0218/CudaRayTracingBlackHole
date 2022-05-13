//
// Created by 王子路 on 2022/5/13.
//

#ifndef BLACKHOLERAYTRACER_IHITABLE_H
#define BLACKHOLERAYTRACER_IHITABLE_H

#include "SchwarzschildBlackHoleEquation.cuh"
#include "src/utiles/ArgbColor.cuh"
#include "vector_types.h"
// IHitable is an abstract class with pure virtual functions, C++'s version of an interface

class IHitable {
 public:
  __host__ IHitable() {}
  __host__ ~IHitable() {}
  __device__ virtual bool Hit(float3 &point, float sqrNorm, float3 &prevPoint, float prevSqrNorm,
							  float3 &velocity, SchwarzschildBlackHoleEquation *equation, float r, float theta,
							  float phi, ArgbColor &color, bool &stop, bool debug, int x, int y) = 0;
 public:
//  __device__ virtual ArgbColor GetColor(int side, float r, float theta, float phi, int x, int y) = 0;
};

#endif