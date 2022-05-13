//
// Created by 王子路 on 2022/5/13.
//
#include "Horizon.cuh"
#include "src/utiles/utils.cuh"
#include "helper_cuda.h"

using namespace std;
using namespace cv;

__host__ Horizon::Horizon(Mat texture, bool checkered) {
  this->checkered = checkered;
  if (!texture.empty()) {
	cout << "horizon texture is not empty" << endl;
	textureWidth = texture.cols;
	textureHeight = texture.rows;
	textureBitmap = texture.clone();
	textureBitmapIsNull = false;
  }
}

__host__ void Horizon::SetAttributes(int gpuId) {
  cudaSetDevice(gpuId);
  if (!textureBitmapIsNull) {
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	const int spitch = textureBitmap.channels() * textureWidth * sizeof(unsigned char);
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;

	printf("horizon texture width:%d\n", textureWidth);
	printf("horizon texture height:%d\n", textureHeight);

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModePoint;
//  texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;
	cudaMallocArray(&array_texture, &channelDesc, textureWidth, textureHeight);
	cudaMemcpy2DToArray(array_texture,
						0,
						0,
						textureBitmap.ptr(0),
						spitch,
						textureBitmap.channels() * textureWidth * sizeof(unsigned char),
						textureHeight,
						cudaMemcpyHostToDevice);
	resDesc.res.array.array = array_texture;
	cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, NULL);
  }

};

__device__ ArgbColor Horizon::GetColor(int side, float r, float theta, float phi, int x, int y) {
  return ArgbColor(0xFF, 0xFF, 0xFF, 0xFF);
}

__host__ void Horizon::ReleaseAttributes(int gpuId) {
  cudaSetDevice(gpuId);
  if (!textureBitmapIsNull) {
	checkCudaErrors(cudaFreeArray(array_texture));
	checkCudaErrors(cudaDestroyTextureObject(tex_obj));
  }
}

__device__ bool Horizon::Hit(float3 &point, float sqrNorm, float3 &prevPoint, float prevSqrNorm,
							 float3 &velocity, SchwarzschildBlackHoleEquation *equation, float r, float theta,
							 float phi, ArgbColor &color, bool &stop, bool debug, int x, int y) {

  // Has the ray fallen past the horizon?
  if (prevSqrNorm > 1 && sqrNorm < 1) {
	float3 colpoint = IntersectionSearch(prevPoint, velocity, equation);

	float tempR = 0., tempTheta = 0., tempPhi = 0.;
	ToSpherical(colpoint, tempR, tempTheta, tempPhi);

	ArgbColor col = ArgbColor(0xFF, 0x00, 0x00, 0x00);
	if (checkered) {
	  float m1 = DoubleMod(tempTheta, 1.04719); // Pi / 3
	  float m2 = DoubleMod(tempPhi, 1.04719); // Pi / 3
	  if ((m1 < 0.52359) ^ (m2 < 0.52359)) { // Pi / 6
		//col = Color.Green
		col = ArgbColor(0xFF, 0x00, 0x80, 0x00);
	  }
	} else if (!textureBitmapIsNull) {
//	  cout << "texturebitmap for horizon not null" << endl;
	  int xPos, yPos;
	  SphericalMap(textureWidth, textureHeight, r, theta, -phi, xPos, yPos);
	  col = ArgbColor::fromArgb(make_uchar4(0, 0, 0, 0));
	}
	color = AddColor(col, color);
	//cout << color << endl;
	stop = true;
	return true;
  }
  return false;
}

__host__ __device__ float3 Horizon::IntersectionSearch(float3 prevPoint, float3 velocity,
													   SchwarzschildBlackHoleEquation *equation) {
  float stepLow = 0, stepHigh = equation->StepSize;
  float3 newPoint = prevPoint;
  float3 tempVelocity;
  while (true) {
	float stepMid = (stepLow + stepHigh) / 2;
	newPoint = prevPoint;
	tempVelocity = velocity;
	equation->Function(newPoint, tempVelocity, stepMid);

	double distance = dot(newPoint, newPoint);
	if (abs(stepHigh - stepLow) < 0.00001) {
	  break;
	}
	if (distance < radius) {
	  stepHigh = stepMid;
	} else {
	  stepLow = stepMid;
	}
  }
  return newPoint;
}
