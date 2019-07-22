#include "kernel.h"
#include "graphic.h"
#define TX 32
#define TY 32

//GPU (커널) 내에서만 호출 가능한 GPU 내부 함수.
__device__ unsigned char clip(int n) {
	return n > 255 ? 255 : (n < 0 ? 0 : n); // 최대 255, 최소 0으로 한정. 색상 범위
}

//cpu에서 호출 가능한, GPU에서 동작하는 커널 함수.
__global__ void distanceKernel(uchar4 *d_out, int w, int h, int2 pos)
{
	const int c = blockIdx.x * blockDim.x + threadIdx.x;	//블록의 id * 블록의 수 + 블록 내에서의 thread id
	const int r = blockIdx.y * blockDim.y + threadIdx.y;	//블록의 id * 블록의 수 + 블록 내에서의 thread id
	//thread는 500*500 개 실행되는게 아니라, 512*512개 실행된다.(TX, TY의 배수) 
	//원하는 범위 밖의 데이터는 계산하지 않는다.
	if ((c >= w) || (r >= h)) return;
	const int i = r * w + c;	//전체 thread 에서의 순서(id)
	//pos 와 자신(c,r) 과의 거리를 계산
	const int dist = sqrtf((c - pos.x)*(c - pos.x) + (r - pos.y)*(r - pos.y));

	//255-d를 최대 255, 최소 0으로 자른다. d가 255 이상이면 0.
	const unsigned char intensity = clip(255 - dist);
	d_out[i].x = intensity;	//R
	d_out[i].y = intensity;	//G
	d_out[i].z = 0;			//B
	d_out[i].w = 255;		//A (불투명)	
}

//커널을 호출하는 CPU 함수. 
void kernelLauncher(uchar4 *d_out, int w, int h, int2 pos) {

	//블록의 크기. 가로가 TX개, 세로가 TY개
	const dim3 blockSize(TX, TY);

	//grid, 즉 thread block의 수. 블록이 grid.x * grid.y 개 존재.
	const dim3 gridSize = dim3((w + TX - 1) / TX, (h + TY - 1) / TY);

	//커널 함수 호출. blockSize 크기의 thread block을, gridSize 만큼 사용한다.
	//공통된 함수 인자로는 GPU 메모리 포인터, 가로, 세로, 기준점
	//distanceKernel << <gridSize, blockSize >> > (d_out, w, h, pos);
}



//cpu에서 호출 가능한, GPU에서 동작하는 커널 함수.
__global__ void starKernel(POS *pos, POS* vel, int w, int h, int screenW,int screenH)
{
	const int c = blockIdx.x * blockDim.x + threadIdx.x;	//블록의 id * 블록의 수 + 블록 내에서의 thread id
	const int r = blockIdx.y * blockDim.y + threadIdx.y;	//블록의 id * 블록의 수 + 블록 내에서의 thread id
	//thread는 500*500 개 실행되는게 아니라, 512*512개 실행된다.(TX, TY의 배수) 
	//원하는 범위 밖의 데이터는 계산하지 않는다.
	if ((c >= w) || (r >= h)) return;
	const int i = r * w + c;	//전체 thread 에서의 순서(id)

	pos[i].x += vel[i].x;
	pos[i].y += vel[i].y;

	if (pos[i].x > screenW || pos[i].x < 0)
		pos[i].x = screenW / 2;
	if (pos[i].y > screenH || pos[i].y < 0)
		pos[i].y = screenH / 2;

	////pos 와 자신(c,r) 과의 거리를 계산
	//const int dist = sqrtf((c - pos.x)*(c - pos.x) + (r - pos.y)*(r - pos.y));

	////255-d를 최대 255, 최소 0으로 자른다. d가 255 이상이면 0.
	//const unsigned char intensity = clip(255 - dist);
	//d_out[i].x = intensity;	//R
	//d_out[i].y = intensity;	//G
	//d_out[i].z = 0;			//B
	//d_out[i].w = 255;		//A (불투명)	
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