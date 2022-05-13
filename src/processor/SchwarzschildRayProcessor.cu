//
// Created by 王子路 on 2022/5/13.
//

#include "vector_types.h"
#include "src/hitables/Sky.cuh"
#include "src/hitables/Horizon.cuh"
#include "src/hitables/TexturedDisk.cuh"
#include "src/utiles/ArgbColor.cuh"
#include "src/utiles/GPUParams.h"
#include "src/utiles/utils.cuh"
#include "helper_math.h"
#include "SchwarzschildRayProcessor.cuh"
#include "cooperative_groups.h"

using namespace cooperative_groups;
namespace cg = cooperative_groups;

#define NUMBERITERATION 10000

__global__ void renderBlackHoleKernel(uchar3 *bitmap,
									  unsigned int linesThisLoop,
									  unsigned int width,
									  unsigned int height,
									  float tanFov,
									  float4 *matrixData,
									  float3 CameraPosition,
									  Sky sky,
									  TexturedDisk texturedDisk,
									  Horizon horizon,
									  SchwarzschildBlackHoleEquation equation,
									  int gpu_id,
									  unsigned int offset) {
  cg::grid_group g = cg::this_grid();
  cg::thread_block b = cg::this_thread_block();
  dim3 grid_index = b.group_index();
  dim3 block_dim = b.group_dim();
  dim3 thread_index = b.thread_index();

  unsigned int gtx = thread_index.x + block_dim.x * grid_index.x;
  unsigned int gty = thread_index.y + block_dim.y * grid_index.y;

  while (gtx < linesThisLoop) {
	gty = thread_index.y + block_dim.y * grid_index.y;
	while (gty < width) {
	  bool debug = false;
	  ArgbColor color;
	  float3 point, prevPoint;
	  float sqrNorm, prevSqrNorm;
	  float tempR = 0, tempTheta = 0, tempPhi = 0;
	  bool stop = false;
	  color = ArgbColor(0x00, 0xFF, 0xFF, 0xFF);

	  float4 view = make_float4((((float)gty) / width - 0.5f) * tanFov,
								((-(float)(gtx + offset) / height + 0.5f) * height / width) * tanFov,
								(float)1.0,
								1.0);
	  view = make_float4(dot(matrixData[0], view),
						 dot(matrixData[1], view),
						 dot(matrixData[2], view),
						 dot(matrixData[3], view));

	  float3 normView = normalize(make_float3(view.x, view.y, view.z));
	  float3 velocity = make_float3(normView.x, normView.y, normView.z);
	  point = CameraPosition;
	  sqrNorm = dot(point, point);
	  stop = false;
	  equation.SetInitialConditions(point, velocity);
	  for (int iter = 0; iter < NUMBERITERATION; iter++) {
		prevPoint = point;
		prevSqrNorm = sqrNorm;
		equation.Function(point, velocity);

		sqrNorm = dot(point, point);
		ToSpherical(point, tempR, tempTheta, tempPhi);


		if (!texturedDisk.Hit(point,
							  sqrNorm,
							  prevPoint,
							  prevSqrNorm,
							  velocity,
							  &equation,
							  tempR,
							  tempTheta,
							  tempPhi,
							  color,
							  stop,
							  debug,
							  gtx,
							  gty)) {
		  if (!horizon.Hit(point,
						   sqrNorm,
						   prevPoint,
						   prevSqrNorm,
						   velocity,
						   &equation,
						   tempR,
						   tempTheta,
						   tempPhi,
						   color,
						   stop,
						   debug,
						   gtx,
						   gty)) {
			sky.Hit(point,
					sqrNorm,
					prevPoint,
					prevSqrNorm,
					velocity,
					&equation,
					tempR,
					tempTheta,
					tempPhi,
					color,
					stop,
					debug,
					gtx,
					gty);
		  }
		}
		if (stop) {
		  break;
		}
	  }

	  if (stop == false) {
		bitmap[gtx * width + gty].x = 0;
		bitmap[gtx * width + gty].y = 0;
		bitmap[gtx * width + gty].z = 0;
	  }
	  bitmap[gtx * width + gty].x = (uchar)color.b;
	  bitmap[gtx * width + gty].y = (uchar)color.g;
	  bitmap[gtx * width + gty].z = (uchar)color.r;

	  gty += g.group_dim().y * b.group_dim().y;
	}
	gtx += g.group_dim().x * b.group_dim().x;
  }
}

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
) {
  dim3 _gridDim(512, 512, 1);
  dim3 _blockDim(16, 16, 1);
  renderBlackHoleKernel<<<_gridDim, _blockDim>>>(bitmap,
												 linesThisLoop,
												 width,
												 height,
												 tanFov,
												 matrixData,
												 CameraPosition,
												 sky,
												 texturedDisk,
												 horizon,
												 equation,
												 gpu_id,
												 offset);
}
