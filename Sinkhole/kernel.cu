#include "kernel.h"
#include "graphic.h"
#define TX 32
#define TY 32

//GPU (Ŀ��) �������� ȣ�� ������ GPU ���� �Լ�.
__device__ unsigned char clip(int n) {
	return n > 255 ? 255 : (n < 0 ? 0 : n); // �ִ� 255, �ּ� 0���� ����. ���� ����
}

//cpu���� ȣ�� ������, GPU���� �����ϴ� Ŀ�� �Լ�.
__global__ void distanceKernel(uchar4 *d_out, int w, int h, int2 pos)
{
	const int c = blockIdx.x * blockDim.x + threadIdx.x;	//����� id * ����� �� + ��� �������� thread id
	const int r = blockIdx.y * blockDim.y + threadIdx.y;	//����� id * ����� �� + ��� �������� thread id
	//thread�� 500*500 �� ����Ǵ°� �ƴ϶�, 512*512�� ����ȴ�.(TX, TY�� ���) 
	//���ϴ� ���� ���� �����ʹ� ������� �ʴ´�.
	if ((c >= w) || (r >= h)) return;
	const int i = r * w + c;	//��ü thread ������ ����(id)
	//pos �� �ڽ�(c,r) ���� �Ÿ��� ���
	const int dist = sqrtf((c - pos.x)*(c - pos.x) + (r - pos.y)*(r - pos.y));

	//255-d�� �ִ� 255, �ּ� 0���� �ڸ���. d�� 255 �̻��̸� 0.
	const unsigned char intensity = clip(255 - dist);
	d_out[i].x = intensity;	//R
	d_out[i].y = intensity;	//G
	d_out[i].z = 0;			//B
	d_out[i].w = 255;		//A (������)	
}

//Ŀ���� ȣ���ϴ� CPU �Լ�. 
void kernelLauncher(uchar4 *d_out, int w, int h, int2 pos) {

	//����� ũ��. ���ΰ� TX��, ���ΰ� TY��
	const dim3 blockSize(TX, TY);

	//grid, �� thread block�� ��. ����� grid.x * grid.y �� ����.
	const dim3 gridSize = dim3((w + TX - 1) / TX, (h + TY - 1) / TY);

	//Ŀ�� �Լ� ȣ��. blockSize ũ���� thread block��, gridSize ��ŭ ����Ѵ�.
	//����� �Լ� ���ڷδ� GPU �޸� ������, ����, ����, ������
	//distanceKernel << <gridSize, blockSize >> > (d_out, w, h, pos);
}



//cpu���� ȣ�� ������, GPU���� �����ϴ� Ŀ�� �Լ�.
__global__ void starKernel(POS *pos, POS* vel, int w, int h, int screenW,int screenH)
{
	const int c = blockIdx.x * blockDim.x + threadIdx.x;	//����� id * ����� �� + ��� �������� thread id
	const int r = blockIdx.y * blockDim.y + threadIdx.y;	//����� id * ����� �� + ��� �������� thread id
	//thread�� 500*500 �� ����Ǵ°� �ƴ϶�, 512*512�� ����ȴ�.(TX, TY�� ���) 
	//���ϴ� ���� ���� �����ʹ� ������� �ʴ´�.
	if ((c >= w) || (r >= h)) return;
	const int i = r * w + c;	//��ü thread ������ ����(id)

	pos[i].x += vel[i].x;
	pos[i].y += vel[i].y;

	if (pos[i].x > screenW || pos[i].x < 0)
		pos[i].x = screenW / 2;
	if (pos[i].y > screenH || pos[i].y < 0)
		pos[i].y = screenH / 2;

	////pos �� �ڽ�(c,r) ���� �Ÿ��� ���
	//const int dist = sqrtf((c - pos.x)*(c - pos.x) + (r - pos.y)*(r - pos.y));

	////255-d�� �ִ� 255, �ּ� 0���� �ڸ���. d�� 255 �̻��̸� 0.
	//const unsigned char intensity = clip(255 - dist);
	//d_out[i].x = intensity;	//R
	//d_out[i].y = intensity;	//G
	//d_out[i].z = 0;			//B
	//d_out[i].w = 255;		//A (������)	
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