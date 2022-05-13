#include "ArgbColor.cuh"

//__device__ ArgbColor ArgbColor::White = ArgbColor(0xFF, 0xFF, 0xFF, 0xFF);
//__constant__ ArgbColor ArgbColor::Black = ArgbColor(0xFF, 0x00, 0x00, 0x00);
//__constant__ ArgbColor ArgbColor::Transparent = ArgbColor(0x00, 0xFF, 0xFF, 0xFF);

__device__ ArgbColor ArgbColor::fromArgb(uchar4 x) {
  ArgbColor c;
  c.b = (uint8_t) x.x;
  c.g = (uint8_t) x.y;
  c.r = (uint8_t) x.z;
  c.a = 0xFF;

  return c;
}