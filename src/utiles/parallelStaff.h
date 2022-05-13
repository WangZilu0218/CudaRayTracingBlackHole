//
// Created by 王子路 on 2021/9/15.
//

#ifndef GOPMCCUDA_PARALLELSTAFF_H
#define GOPMCCUDA_PARALLELSTAFF_H
//#include "Macro.h"
//#include "helper_cuda.h"

#include "omp.h"
#include "stdio.h"
#include "time.h"

class parallelStaff {
 public:
  parallelStaff ();
  ~parallelStaff ();
  int getNumGPUs() const { return num_gpus; }
//  unsigned int nBatch() { return GRIDDIM * BLOCKDIM; }
 private:
  int num_gpus;
  //unsigned int nBatch;
};


#endif //GOPMCCUDA_PARALLELSTAFF_H
