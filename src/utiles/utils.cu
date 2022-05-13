#include "utils.cuh"

__host__ __device__ void ToCartesian(const float r, const float theta, const float phi,
									 float &x, float &y, float &z) {
  x = r * cos(phi) * sin(theta);
  y = r * sin(phi) * sin(theta);
  z = r * cos(theta);
}

__host__ __device__ void ToSpherical(const float x, const float y, const float z,
									 float &r, float &theta, float &phi) {
  r = sqrt(x * x + y * y + z * z);
  phi = atan2(y, x);
  theta = acos(z / r);
}

__host__ __device__ void ToSpherical(const float3 v, float &r,
									 float &theta, float &phi) {
  r = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  phi = atan2(v.y, v.x);
  theta = acos(v.z / r);
}

__host__ __device__ float DoubleMod(float n, float m) {
  float x = floor(n / m);
  return n - (m * x);
}

__host__ __device__ float GetBrightness(ArgbColor c) {
  float r = (float)c.r / 255.0f;
  float g = (float)c.g / 255.0f;
  float b = (float)c.b / 255.0f;

  float max, min;

  max = r;
  min = r;

  if (g > max) max = g;
  if (b > max) max = b;

  if (g < min) min = g;
  if (b < min) min = b;

  return (max + min) / 2.0;
}

__host__ __device__ int Cap(int x, int max) {
  if (x > max) {
	return max;
  } else {
	return x;
  }
}

__host__ __device__ int CapMin(int x, int min) {
  if (x < min) {
	return min;
  } else {
	return x;
  }
}

__host__ __device__ ArgbColor AddColor(ArgbColor hitColor, ArgbColor tintColor) {
  if (tintColor == ArgbColor(0x00, 0xFF, 0xFF, 0xFF)) {
	return hitColor;
  }
  float brightness = GetBrightness(tintColor);
  ArgbColor c;
  c.r = (uint8_t)Cap((int)(((1.0 - brightness) * hitColor.r) +
	  CapMin(tintColor.r, 0) * 255 / 205
  ), 255);
  c.g = (uint8_t)Cap((int)(((1.0 - brightness) * hitColor.g) +
	  CapMin(tintColor.g, 0) * 255 / 205
  ), 255);
  c.b = (uint8_t)Cap((int)(((1.0 - brightness) * hitColor.b) +
	  CapMin(tintColor.b, 0) * 255 / 205
  ), 255);
  c.a = (uint8_t)0xFF;
  return c;
}

__host__ __device__ void SphericalMap(int SizeX,
									  int SizeY,
									  const float r,
									  const float theta,
									  const float phi,
									  int &x,
									  int &y) {
  x = (int)((phi / (2.0 * M_PI)) * SizeX) % SizeX;
  y = (int)((theta / M_PI) * SizeY) % SizeY;
  if (x < 0) { x = SizeX + x; }
  if (y < 0) { y = SizeY + y; }
}

__host__ __device__ void DiskMap(float rMin,
								 float rMax,
								 int SizeX,
								 int SizeY,
								 const float r,
								 const float theta,
								 const float phi,
								 int &x,
								 int &y) {
  if (r < rMin || r > rMax) {
	x = 0;
	y = SizeY;
  }

  x = (int)((phi / (2 * M_PI)) * SizeX) % SizeX;
  if (x < 0) { x = SizeX + x; }
  y = (int)(((r - rMin) / (rMax - rMin)) * SizeY);
  if (y > SizeY - 1) { y = SizeY - 1; }
}

__host__ cv::Mat getNativeTextureBitmap(cv::Mat texture) {
  return texture.clone();
}
