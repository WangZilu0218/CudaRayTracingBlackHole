//
// Created by 王子路 on 2022/5/13.
//
#include <iostream>
#include <stdio.h>
#include <string>
#include "vector"
#include "src/utiles/parallelStaff.h"
#include "opencv2/opencv.hpp"
#include "vector_types.h"

#include "src/hitables/IHitable.cuh"
#include "src/hitables/TexturedDisk.cuh"
#include "src/hitables/Horizon.cuh"
#include "src/hitables/Sky.cuh"
#include "src/Scene.h"
#include "src/processor/SchwarzschildRayProcessor.h"

#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/numpy.h"

#include "helper_cuda.h"

using namespace std;
using namespace cv;

int renderBlackHoleOneFrame(pybind11::array cameraPosition,
							pybind11::array cameraLookAt,
							pybind11::array cameraUp,
							int frame) {
  auto hCameraPosition = pybind11::cast<pybind11::array_t<float>>(cameraPosition).request();
  auto hCameraLookAt = pybind11::cast<pybind11::array_t<float>>(cameraLookAt).request();
  auto hCameraUp = pybind11::cast<pybind11::array_t<float>>(cameraUp).request();

  float3 cameraPos, lookAt, up;
  cameraPos = make_float3(((float *)hCameraPosition.ptr)[0],
						  ((float *)hCameraPosition.ptr)[1],
						  ((float *)hCameraPosition.ptr)[2]);

  printf("camera pos x:%f y:%f z:%f\n", cameraPos.x, cameraPos.y, cameraPos.z);
  lookAt = make_float3(((float *)hCameraLookAt.ptr)[0],
					   ((float *)hCameraLookAt.ptr)[1],
					   ((float *)hCameraLookAt.ptr)[2]);

  up = make_float3(((float *)hCameraUp.ptr)[0],
				   ((float *)hCameraUp.ptr)[1],
				   ((float *)hCameraUp.ptr)[2]);

  printf("Frame:%d\n", frame);

  parallelStaff staff;
  int frame_number = 1;
  checkCudaErrors(cudaFree(0));
  unsigned int num_gpus = staff.getNumGPUs();
//  float3 cameraPos = make_float3(0, 2.1221160500310874, 29.924849598121632);
//  float3 lookAt = make_float3(0, 0, 0);
//  float3 up = make_float3(-0.3, 1, 0);
  float fov = 55.0;
  float curvatureCoeff = -1.5;
  float angularMomentum = 0.0;
  string fileName = "out";
  fileName += to_string(frame);
  fileName += ".jpg";
  Mat diskImg = imread("../images/starless_disk.jpg", 1);
  Mat skyImg = imread("../images/sky8k.jpg", 1);
  Mat diskImgBmp, skyImgBmp;
  diskImg.convertTo(diskImgBmp, CV_8UC3);
  skyImg.convertTo(skyImgBmp, CV_8UC3);
  Mat horzTexture;
  cv::Mat newSkyMap(skyImgBmp.size(), CV_MAKE_TYPE(skyImgBmp.depth(), 4));
  cv::Mat newDiskMap(diskImgBmp.size(), CV_MAKE_TYPE(diskImgBmp.depth(), 4));
  int from_to[] = {0, 0, 1, 1, 2, 2, -1, 3};
  cv::mixChannels(&skyImgBmp, 1, &newSkyMap, 1, from_to, 4);
  cv::mixChannels(&diskImgBmp, 1, &newDiskMap, 1, from_to, 4);

  std::vector<Sky> sky_list;
  std::vector<Horizon> horizon_list;
  std::vector<TexturedDisk> textureDisk_list;

  for (int i = 0; i < num_gpus; i++) {
	Sky sky = Sky(newSkyMap, 30);
	sky.SetTextureOffset(M_PI / 2);
	sky_list.push_back(sky);
	Horizon horizon = Horizon(horzTexture, false);
	horizon_list.push_back(horizon);
	TexturedDisk texturedDisk = TexturedDisk(2.6, 12.0, newDiskMap);
	textureDisk_list.push_back(texturedDisk);
  }

  Scene *scene = new Scene(staff,
						   cameraPos,
						   lookAt,
						   up,
						   fov,
						   sky_list,
						   horizon_list,
						   textureDisk_list,
						   curvatureCoeff,
						   angularMomentum);
  scene->SetDeviceMatrixData();
  scene->SetHitableData();
  SchwarzschildRayProcessor(1280 * 2, 720 * 2, scene, fileName).Process(staff, curvatureCoeff);

  scene->ReleaseDeviceMatrixDate();
  scene->ReleaseHitableData();
  return 0;
}

PYBIND11_MODULE(RayTraceBlackHole, m) {
  m.def("renderBlackHoleOneFrame", renderBlackHoleOneFrame);
}