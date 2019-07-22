#include "kernel.h"
#include "graphic.h"
#define TX 32
#define TY 32

__device__ float2 vectorScale(float2 vec, float size) {
	float s = vec.x * vec.x + vec.y*vec.y;
	s = sqrt(s);
	vec.x = vec.x / s * size;
	vec.y = vec.y / s * size;

	return vec;
}

__device__ float2 velocity(POS pos, POS speed, int screenW, int screenH) {
	float2 vel;

	float2 vec, gravity;
	
		float x, y, c;
		x = pos.x - screenW / 2;
		y = pos.y - screenH / 2;

		gravity.x = -x; gravity.y = -y;
		gravity = vectorScale(gravity, speed.y);

		//원점을 중심으로 직각으로 움직이는 방향의 벡터(x축이나 y축까지 이동)
		if (x == 0) {//y축
			vec.x = speed.x;
			if (y > 0)
				vec.x *= -1;
			vec.y = 0;
		}
		else if (y == 0) {//x축
			vec.x = 0;
			vec.y = speed.x;
			if (x > 0)
				vec.y *= -1;
		}
		else if (x > 0 && y > 0 || x < 0 && y < 0) {//1, 3사분면. (c,0)
			c = (x * x + y * y) / x;
			vec.x = c - x;
			vec.y = -y;
			vec = vectorScale(vec, speed.x);
		}
		else if (x > 0 && y < 0 || x < 0 && y > 0) {//2, 4사분면. (0,c),
			c = (x * x + y * y) / y;
			vec.x = -x;
			vec.y = c - y;
			vec = vectorScale(vec, speed.x);
		}
	

	//if (x < speed.x && y < speed.x && x > -speed.x && y > -speed.x) {
	//	gravity.x *= -100;
	//	gravity.y *= -100;
	//}

	vel.x = vec.x + gravity.x;
	vel.y = vec.y + gravity.y;

	return vel;
}

//cpu에서 호출 가능한, GPU에서 동작하는 커널 함수.
__global__ void starKernel(POS *pos, POS* speed, int w, int h, int screenW, int screenH)
{
	const int c = blockIdx.x * blockDim.x + threadIdx.x;	//블록의 id * 블록의 수 + 블록 내에서의 thread id
	const int r = blockIdx.y * blockDim.y + threadIdx.y;	//블록의 id * 블록의 수 + 블록 내에서의 thread id
	//thread는 500*500 개 실행되는게 아니라, 512*512개 실행된다.(TX, TY의 배수) 
	//원하는 범위 밖의 데이터는 계산하지 않는다.
	if ((c >= w) || (r >= h)) return;
	const int i = r * w + c;	//전체 thread 에서의 순서(id)

	float2 vel = velocity(pos[i],speed[i], screenW, screenH);
	

	pos[i].x += vel.x ;
	pos[i].y += vel.y ;

	/*pos[i].x += vel[i].x;
	pos[i].y += vel[i].y;

	if (pos[i].x > screenW || pos[i].x < 0)
		pos[i].x = screenW / 2;
	if (pos[i].y > screenH || pos[i].y < 0)
		pos[i].y = screenH / 2;*/

}

//커널을 호출하는 CPU 함수. 
void kernelLauncher_star(POS *pos, POS* vel, int w, int h, int screenW, int screenH) {

	

	//블록의 크기. 가로가 TX개, 세로가 TY개
	const dim3 blockSize(TX, TY);

	//grid, 즉 thread block의 수. 블록이 grid.x * grid.y 개 존재.
	const dim3 gridSize = dim3((w + TX - 1) / TX, (h + TY - 1) / TY);

	//커널 함수 호출. blockSize 크기의 thread block을, gridSize 만큼 사용한다.
	//공통된 함수 인자로는 GPU 메모리 포인터, 가로, 세로, 기준점
	//starKernel << <gridSize, blockSize >> > (pos, vel, w, h);
	starKernel KERNEL_ARGS2(gridSize, blockSize) (pos, vel, w, h, screenW , screenH);
}