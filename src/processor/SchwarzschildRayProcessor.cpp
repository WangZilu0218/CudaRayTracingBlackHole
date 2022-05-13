//
// Created by 王子路 on 2022/5/13.
//

#include <iostream>
#include <thread>
#include <ctime>
#include "omp.h"

#include "SchwarzschildRayProcessor.h"
#include "src/hitables/SchwarzschildBlackHoleEquation.cuh"
#include "SchwarzschildRayProcessor.cuh"
#include "src/Scene.h"
#include "helper_cuda.h"

using namespace cv;

SchwarzschildRayProcessor::SchwarzschildRayProcessor(int width, int height,
													 Scene *scene, string outputFileName) {
  this->width = width;
  this->height = height;
  this->scene = scene;
  this->outputFileName = outputFileName;
}

void SchwarzschildRayProcessor::Process(parallelStaff &staff, float curvatureCoeff) {
  // Create main bitmap for writing pixels

  // Primitive type defined in the form
  // CV_<bit-depth>{U|S|F}C(<number_of_channels>)
  // U = unsigned integer, S = signed integer, F = float
  // https://stackoverflow.com/questions/27183946/what-does-cv-8uc3-and-the-other-types-stand-for-in-opencv
  outputBitmap = Mat(height, width, CV_8UC3);  // rows, cols, type
  // CV_Assert(outputBitmap.channels() == 4);

//  int numThreads = 8;
  time_t now = time(0);
  unsigned int numDevices;
  numDevices = staff.getNumGPUs();
//  cout << "Launching " << numThreads << " threads..." << endl;

  std::vector<std::vector<int> *> lineLists;
  size_t free = 0;
  size_t total = 0;
  cudaSetDevice(0);
  checkCudaErrors(cudaMemGetInfo(&free, &total));
  std::vector<GPUParams> paramList;
  int offset = 0;
  int deviceCount = numDevices;
  for (int i = 0; i < numDevices; i++) {
	checkCudaErrors(cudaSetDevice(i));
	GPUParams gp;
	unsigned int maxLinesOnThisDevice = (unsigned int)(free * 0.8 / (width * sizeof(uchar4)));
	unsigned int numLinesThisDevice = (int)ceil((float)height / (float)numDevices);
	gp.hBitmap = outputBitmap.ptr<uchar3>(0) + offset * width;
	gp.linesThisDevice = numLinesThisDevice;
	gp.gpuID = i;
	gp.offset = offset;
	if (numLinesThisDevice > maxLinesOnThisDevice) {
	  gp.linesOneLoop = maxLinesOnThisDevice;
	} else {
	  gp.linesOneLoop = numLinesThisDevice;
	}
	paramList.push_back(gp);
	offset += numLinesThisDevice;
	deviceCount--;
  }
  omp_set_dynamic(0);

#pragma omp parallel num_threads(numDevices)
  {
    SchwarzschildBlackHoleEquation equation(curvatureCoeff);
    int tid = omp_get_thread_num();
	RayTraceThread(paramList[tid], equation);
  }
#pragma omp barrier
  imwrite(outputFileName, outputBitmap);
  cout << "Finished in " << time(0) - now << " seconds." << endl;
}

void SchwarzschildRayProcessor::RayTraceThread(GPUParams &threadParam, SchwarzschildBlackHoleEquation equation) {
  checkCudaErrors(cudaSetDevice(threadParam.gpuID));
  for (int i = 0; i < threadParam.linesThisDevice; i += threadParam.linesOneLoop) {
	uchar3 *dBitmap;
	int linesLeft = threadParam.linesThisDevice - i;
	int linesThisLoop = 0;
	if (linesLeft > threadParam.linesOneLoop){
	  linesThisLoop = threadParam.linesOneLoop;
	} else {
	  linesThisLoop = linesLeft;
	}
	checkCudaErrors(cudaMalloc((void **)&dBitmap, sizeof(uchar3) * linesThisLoop * width));
	renderBlackHole(dBitmap,
					linesThisLoop,
					width,
					height,
					scene->tanFov,
					scene->dMatrixData_list[threadParam.gpuID],
					scene->CameraPosition,
					scene->sky_list[threadParam.gpuID],
					scene->textureDisk_list[threadParam.gpuID],
					scene->horizon_list[threadParam.gpuID],
					equation,
					threadParam.gpuID,
					threadParam.offset);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(threadParam.hBitmap + i,
							   dBitmap,
							   sizeof(uchar3) * linesLeft * width,
							   cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(dBitmap));
  }
}
