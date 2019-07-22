#pragma once
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

typedef struct _POS {
	float x, y;

}POS;

void initialStar(int size, int w, int h);
void drawStars();
void setup(int W, int H, int screenW, int screenH);

