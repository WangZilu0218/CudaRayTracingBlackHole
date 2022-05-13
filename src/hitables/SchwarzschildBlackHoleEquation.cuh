//
// Created by 王子路 on 2022/5/13.
//

#ifndef BLACKHOLERAYTRACER_SCHWARZSCHILDBLACKHOLEEQUATION_H
#define BLACKHOLERAYTRACER_SCHWARZSCHILDBLACKHOLEEQUATION_H

//#include "models/Vector3D.h"
#include "vector_types.h"
#include "helper_math.h"
#include <vector>


//using namespace Models;

struct SchwarzschildBlackHoleEquation {
  const float DefaultStepSize = 0.16f;

  float h2;
  float StepSize;

  // Multiplier for the potential
  // ranging from 0 for no curvature, to -1.5 for full curvature
  float PotentialCoefficient;

  __host__ explicit SchwarzschildBlackHoleEquation(float potentialCoef) {
	PotentialCoefficient = potentialCoef;
	StepSize = DefaultStepSize;
  }

  __host__ SchwarzschildBlackHoleEquation(SchwarzschildBlackHoleEquation &other) {
	this->StepSize = other.StepSize;
	this->PotentialCoefficient = other.PotentialCoefficient;
  }

  // Note: this function marked as unsafe in c# code
  __host__ __device__ void SetInitialConditions(float3 &point, float3 &velocity) {
	float3 c = cross(point, velocity);
	h2 = dot(c, c);
  }

  __host__ __device__ void Function(float3 &point, float3 &velocity) {
	Function(point, velocity, (sqrtf(dot(point, point)) / 30.0) * StepSize);
  }

  __host__ __device__ void Function(float3 &point, float3 &velocity, float step) {
	point += velocity * step;

	// this is the magical - 3/2 r^(-5) potential...
	float3 accel = PotentialCoefficient * h2 *
		point / (float)pow(dot(point, point), 2.5);
	velocity += accel * step;
  }
};


#endif  // BLACKHOLERAYTRACER_SCHWARZSCHILDBLACKHOLEEQUATION_H
