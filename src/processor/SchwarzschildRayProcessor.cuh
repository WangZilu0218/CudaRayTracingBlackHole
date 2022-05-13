//
// Created by 王子路 on 2022/5/13.
//
#include "src/utiles/GPUParams.h"

void renderBlackHole(uchar3 *bitmap,
					 unsigned int linesThisLoop,
					 unsigned int width,
					 unsigned int height,
					 float tanFov,
					 float4 *matrixData,
					 float3 CameraPosition,
					 Sky &sky,
					 TexturedDisk &texturedDisk,
					 Horizon &horizon,
					 SchwarzschildBlackHoleEquation equation,
					 int gpu_id,
					 unsigned int offset
);