#ifndef BLACKHOLERAYTRACER_ARGBCOLOR_H
#define BLACKHOLERAYTRACER_ARGBCOLOR_H
#include "opencv2/opencv.hpp"

class ArgbColor {
 public:
//  uchar4 channel;
  uint8_t a;  // alpha channel
  uint8_t r;  // < value of red chanel
  uint8_t g;  // < value of green chanel
  uint8_t b;  // < value of blue chanel


  __host__ __device__ ArgbColor(uint8_t a = 255, uint8_t r = 0,
			uint8_t g = 0, uint8_t b = 0) : r(r), g(g), b(b), a(a) { }

  // const dog
  __host__ friend std::ostream &operator<<(std::ostream &strm, const ArgbColor &c);

  // comparison
  __host__ __device__ inline bool operator==(const ArgbColor &rhs) const {
	return a == rhs.a && r == rhs.r && g == rhs.g && b == rhs.b;
  }

  __host__ __device__ inline bool operator!=(const ArgbColor &rhs) const {
	return !operator==(rhs);
  }

//  __host__ __device__ ArgbColor fromArgb(uchar4 x) {
//	ArgbColor c;
//	c.b = (uint8_t) x.x;
//	c.g = (uint8_t) x.y;
//	c.r = (uint8_t) x.z;
//	c.a = 0xFF;
//
//	return c;
//  }
  /*
  Construct a ArgbColor object from a uint32_t in argb format
  https://en.wikipedia.org/wiki/RGBA_color_space#ARGB_(word-order)
  */
  __device__ static ArgbColor fromArgb(uchar4);
};

#endif  // BLACKHOLERAYTRACER_ARGBCOLOR_H