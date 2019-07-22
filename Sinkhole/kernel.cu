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

		//������ �߽����� �������� �����̴� ������ ����(x���̳� y����� �̵�)
		if (x == 0) {//y��
			vec.x = speed.x;
			if (y > 0)
				vec.x *= -1;
			vec.y = 0;
		}
		else if (y == 0) {//x��
			vec.x = 0;
			vec.y = speed.x;
			if (x > 0)
				vec.y *= -1;
		}
		else if (x > 0 && y > 0 || x < 0 && y < 0) {//1, 3��и�. (c,0)
			c = (x * x + y * y) / x;
			vec.x = c - x;
			vec.y = -y;
			vec = vectorScale(vec, speed.x);
		}
		else if (x > 0 && y < 0 || x < 0 && y > 0) {//2, 4��и�. (0,c),
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

//cpu���� ȣ�� ������, GPU���� �����ϴ� Ŀ�� �Լ�.
__global__ void starKernel(POS *pos, POS* speed, int w, int h, int screenW, int screenH)
{
	const int c = blockIdx.x * blockDim.x + threadIdx.x;	//����� id * ����� �� + ��� �������� thread id
	const int r = blockIdx.y * blockDim.y + threadIdx.y;	//����� id * ����� �� + ��� �������� thread id
	//thread�� 500*500 �� ����Ǵ°� �ƴ϶�, 512*512�� ����ȴ�.(TX, TY�� ���) 
	//���ϴ� ���� ���� �����ʹ� ������� �ʴ´�.
	if ((c >= w) || (r >= h)) return;
	const int i = r * w + c;	//��ü thread ������ ����(id)

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

//Ŀ���� ȣ���ϴ� CPU �Լ�. 
void kernelLauncher_star(POS *pos, POS* vel, int w, int h, int screenW, int screenH) {

	

	//����� ũ��. ���ΰ� TX��, ���ΰ� TY��
	const dim3 blockSize(TX, TY);

	//grid, �� thread block�� ��. ����� grid.x * grid.y �� ����.
	const dim3 gridSize = dim3((w + TX - 1) / TX, (h + TY - 1) / TY);

	//Ŀ�� �Լ� ȣ��. blockSize ũ���� thread block��, gridSize ��ŭ ����Ѵ�.
	//����� �Լ� ���ڷδ� GPU �޸� ������, ����, ����, ������
	//starKernel << <gridSize, blockSize >> > (pos, vel, w, h);
	starKernel KERNEL_ARGS2(gridSize, blockSize) (pos, vel, w, h, screenW , screenH);
}