//
// Created by 王子路 on 2021/9/15.
//
#include "cuda.h"
#include "cuda_runtime.h"
#include "parallelStaff.h"

parallelStaff::parallelStaff ()
{
  srand (time (0));
  cudaGetDeviceCount (&num_gpus);
  if (num_gpus < 1)
  {
	printf ("no CUDA capable devices were detected\n");
	throw -1;
  }
  printf ("number of host CPUs:\t%d\n", omp_get_num_procs ());
  printf ("number of CUDA devices:\t%d\n", num_gpus);
  for (int i = 0; i < num_gpus; i++)
  {
	cudaDeviceProp dprop;
	cudaGetDeviceProperties (&dprop, i);
	printf ("   %d: %s\n", i, dprop.name);
	if (!dprop.deviceOverlap)
	{
	  printf ("Device will not handle overlaps, so no speed up from streams\n");
	}
  }

  printf ("---------------------------\n");
//  omp_set_dynamic (0);
//  omp_set_num_threads (num_gpus);  // create as many CPU threads as there are CUDA devices
}

parallelStaff::~parallelStaff ()
{

}