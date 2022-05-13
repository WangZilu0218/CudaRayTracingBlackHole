//
// Created by 王子路 on 2022/4/29.
//

#ifndef RAYTRACEBLACKHOLECUDA__SCENE_H_
#define RAYTRACEBLACKHOLECUDA__SCENE_H_
#ifndef SCENE_H
#define SCENE_H

#include "hitables/IHitable.cuh"
#include "vector_types.h"
#include <opencv2/opencv.hpp>
#include <iostream>

#include "hitables/Sky.cuh"
#include "hitables/Horizon.cuh"
#include "hitables/TexturedDisk.cuh"
#include "utiles/parallelStaff.h"

//using namespace Models;
using namespace std;
using namespace cv;

class Scene {
 public:
  float3 CameraPosition;
  float3 CameraLookAt;
  float3 UpVector;
  float3 front;
  float3 left;
  float3 nUp;

  float Fov;
  float4 *matrixData;
  float tanFov;
  float CameraDistance;
  float CameraAngleHorz;
  float CameraAngleVert;
  float CameraTilt;

  std::vector<Sky> sky_list;
  std::vector<Horizon> horizon_list;
  std::vector<TexturedDisk> textureDisk_list;
  std::vector<float4 *> dMatrixData_list;

  const parallelStaff &staff;


  __host__ Scene(const parallelStaff &staff,
				 float3 CameraPosition,
				 float3 CameraLookAt,
				 float3 UpVector,
				 float Fov,
				 std::vector<Sky> sky_list,
				 std::vector<Horizon> horizon_list,
				 std::vector<TexturedDisk> textureDisk_list,
				 float CurvatureCoeff,
				 float AngularMomentum);

  __host__ void SetHitableData();
  __host__ void ReleaseHitableData();
  __host__ void SetDeviceMatrixData();
  __host__ void ReleaseDeviceMatrixDate();
  __host__ ~Scene();
};

#endif  // SCENE_H

#endif //RAYTRACEBLACKHOLECUDA__SCENE_H_
