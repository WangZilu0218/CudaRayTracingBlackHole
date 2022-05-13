//
// Created by 王子路 on 2022/5/13.
//
#include "Scene.h"
#include "utiles/utils.cuh"
#include "helper_cuda.h"

__host__ Scene::Scene(const parallelStaff &staff,
					  float3 CameraPosition,
					  float3 CameraLookAt,
					  float3 UpVector,
					  float Fov,
					  std::vector<Sky> sky_list,
					  std::vector<Horizon> horizon_list,
					  std::vector<TexturedDisk> textureDisk_list,
					  float CurvatureCoeff,
					  float AngularMomentum)
	: staff(staff), sky_list(sky_list), horizon_list(horizon_list), textureDisk_list(textureDisk_list) {
  this->CameraPosition = CameraPosition;
  this->CameraLookAt = CameraLookAt;
  this->UpVector = UpVector;
  this->Fov = Fov;

  float tempR = 0;
  float tempTheta = 0;
  float tempPhi = 0;

  ToSpherical(CameraPosition, tempR, tempTheta, tempPhi);
  this->CameraDistance = tempR;
  this->CameraAngleVert = tempTheta;
  this->CameraAngleHorz = tempPhi - 0.1;
  this->tanFov = (float)tan((M_PI / 180.0) * Fov);

  this->front = normalize(CameraLookAt - CameraPosition);
  this->left = normalize(cross(UpVector, front));
  this->nUp = cross(front, left);
  this->matrixData = new float4[4];
  matrixData[0].x = left.x;
  matrixData[0].y = nUp.x;
  matrixData[0].z = front.x;
  matrixData[0].w = 0.0;
  matrixData[1].x = left.y;
  matrixData[1].y = nUp.y;
  matrixData[1].z = front.y;
  matrixData[1].w = 0.0;
  matrixData[2].x = left.z;
  matrixData[2].y = nUp.z;
  matrixData[2].z = front.z;
  matrixData[2].w = 0.0;
  matrixData[3].x = 0.0;
  matrixData[3].y = 0.0;
  matrixData[3].z = 0.0;
  matrixData[3].w = 0.0;
}

__host__ void Scene::SetHitableData() {
  for (int i = 0; i < staff.getNumGPUs(); i++) {
	sky_list[i].SetAttributes(i);
	horizon_list[i].SetAttributes(i);
	textureDisk_list[i].SetAttributes(i);
  }
}

__host__ void Scene::ReleaseHitableData() {
  for (int i = 0; i < staff.getNumGPUs(); i++) {
	sky_list[i].ReleaseAttributes(i);
	horizon_list[i].ReleaseAttributes(i);
	textureDisk_list[i].ReleaseAtrributes(i);
  }
}

__host__ void Scene::SetDeviceMatrixData() {
  for(int i = 0;i<staff.getNumGPUs();i++){
	cudaSetDevice(i);
	float4 *d_matrixData;
	checkCudaErrors(cudaMalloc((void **)&d_matrixData, 16 * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_matrixData, matrixData, 16 * sizeof(float), cudaMemcpyHostToDevice));
	dMatrixData_list.push_back(d_matrixData);
  }
}

__host__ void Scene::ReleaseDeviceMatrixDate() {
  for(int i=0;i<staff.getNumGPUs();i++){
	cudaSetDevice(i);
	checkCudaErrors(cudaFree(dMatrixData_list[i]));
  }
}

__host__ Scene::~Scene() {
  delete[] matrixData;
}