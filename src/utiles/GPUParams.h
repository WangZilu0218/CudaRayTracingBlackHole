//
// Created by 王子路 on 2022/5/1.
//

#ifndef RAYTRACEBLACKHOLECUDA__GPUPARAMS_H_
#define RAYTRACEBLACKHOLECUDA__GPUPARAMS_H_
#include "vector_types.h"
class GPUParams {
 public:
  unsigned int gpuID;
  unsigned int linesOneLoop;
  unsigned int linesThisDevice;
  unsigned int offset;
//  SchwarzschildBlackHoleEquation *Equation;
  uchar3 *hBitmap;
};

#endif //RAYTRACEBLACKHOLECUDA__GPUPARAMS_H_
