//
// Created by 王子路 on 2022/5/13.
//

#ifndef BLACKHOLERAYTRACER_SCHWARZSCHILDRAYPROCESSOR_H
#define BLACKHOLERAYTRACER_SCHWARZSCHILDRAYPROCESSOR_H

//#include "SchwarzschildBlackHoleEquation.cuh"
#include <opencv2/opencv.hpp>
#include <iostream>
#include "src/utiles/GPUParams.h"
#include "src/utiles/parallelStaff.h"
#include "src/Scene.h"

//using namespace Models;
using namespace std;
using namespace cv;
//namespace BlackHoleRaytracer {

class SchwarzschildRayProcessor {
 private:
  int width;
  int height;
  Scene *scene;

  Mat outputBitmap;
  String outputFileName;
  const int NumIterations = 10000;
//  std::vector<std::thread*> workerThreads;  ///< pool of worker threads


 public:
  SchwarzschildRayProcessor(int width, int height,
							Scene *scene, string outputFileName);
  void Process(parallelStaff &staff, float curvatureCoeff);
  void RayTraceThread(GPUParams &threadParams, SchwarzschildBlackHoleEquation equation);
};

//}  // namespace BlackHoleRaytracer

#endif  // BLACKHOLERAYTRACER_SCHWARZSCHILDRAYPROCESSOR_H
