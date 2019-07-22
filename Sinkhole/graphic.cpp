#include "graphic.h"
#include "kernel.h"
#include <stdio.h>

#define MINWIDTH 0
#define MINHEIGHT 0
#define MAXWIDTH 1024
#define MAXHEIGHT 1024

#define MINSPEED 1
#define MAXSPEED 1

struct cudaGraphicsResource *cuda_pos_resource;	//openGL의 pixel buffer와 연결하기 위한 변수.
struct cudaGraphicsResource *cuda_vel_resource;	//openGL의 pixel buffer와 연결하기 위한 변수.
GLuint pos_buf, vel_buf;
int star_size;
int bufferSIze;

void normalize(POS& pos) {

	float size = pos.x*pos.x + pos.y*pos.y;
	size = sqrt(size);

	pos.x = pos.x / size;
	pos.y = pos.y / size;

}

void posSizeFIx(POS& pos, float size) {
	normalize(pos);
	pos.x *= size;
	pos.y *= size;
}

void initialStar(int size, int w, int h) {
	star_size = size;
	bufferSIze = 2 * size * sizeof(GLfloat); //x,y

	GLfloat* init_pos = (GLfloat*)malloc(bufferSIze);	//x,y

	int pos_idx = 0;
	for (int i = 0; i < size; i++) {
		init_pos[pos_idx++] = rand() % (MAXWIDTH - MINWIDTH);		//x
		init_pos[pos_idx++] = rand() % (MAXHEIGHT - MINHEIGHT);		//y
		//printf("[%d] pos : %f,%f\n", i, init_pos[pos_idx - 2], init_pos[pos_idx - 1]);
	}

	init_pos[0] = 3.2 + w / 2;
	init_pos[1] = 2.4 + h / 2;



	float speed = (rand() / (float)RAND_MAX) * (float)(MAXSPEED - MINSPEED) + MINSPEED;

	GLfloat* init_vel = (GLfloat*)malloc(bufferSIze);	//x,y
	POS mid;	//기준점, 중점.
	mid.x = w / 2; mid.y = h / 2;

	float angle;
	pos_idx = 0;
	int vel_idx = 0;
	for (int i = 0; i < size; i++) {

	/*	float x, y;
		x = init_pos[pos_idx++] - mid.x;
		y = init_pos[pos_idx++] - mid.y;

		if (x >= 0) {
			init_vel[vel_idx++] = 1;	
		}
		else {
			init_vel[vel_idx++] = -1;
		}


		if (y >= 0) {
			init_vel[vel_idx++] = 1;
		}
		else {
			init_vel[vel_idx++] = -1;
		}
*/
		speed = (rand() / (float)RAND_MAX) * (float)(MAXSPEED - MINSPEED) + MINSPEED;	//속력. 순수하게 크기

		speed = 2;
		
		float x, y, l, theta, c;
		POS vec;
		x = init_pos[pos_idx++] - mid.x;
		y = init_pos[pos_idx++] - mid.y;

		//원점을 중심으로 직각으로 움직이는 방향의 벡터(x축이나 y축까지 이동)
		if (x == 0) {//y축
			vec.x = speed;
			if (y > 0)
				vec.x *= -1;
			vec.y = 0;
		}
		else if (y == 0) {//x축
			vec.x = 0;
			vec.y = speed;
			if (x > 0)
				vec.y *= -1;
		}
		else if (x > 0 && y > 0 || x < 0 && y < 0) {//1, 3사분면. (c,0)
			c = (x * x + y * y) / x;
			vec.x = c-x;
			vec.y = -y;
			posSizeFIx(vec, speed);
		}
		else if (x > 0 && y < 0 || x < 0 && y > 0) {//2, 4사분면. (0,c),
			c = (x * x + y * y) / y;
			vec.x = -x;
			vec.y = c-y;
			posSizeFIx(vec, speed);

		}
		else {
			vec.x = 0;
			vec.y = 0;
		}

		//l = sqrt(x*x + y * y);
		//theta = acos(x / l);

		init_vel[vel_idx++] = vec.x;
		init_vel[vel_idx++] = vec.y;

		//init_vel[vel_idx++] = speed;
		//init_vel[vel_idx++] = speed;
		
	//	printf("[%d] vel : %f,%f\n",i, init_vel[vel_idx-2], init_vel[vel_idx-1]);
	}

	//for (int i = 0; i < size; i++) {
	//	printf("%d [%f,%f] - (%f,%f)\n",i, init_pos[i*2], init_pos[i*2+1], init_vel[i*2], init_vel[i*2+1]);
	//}


	glGenBuffers(1, &pos_buf);

	glBindBuffer(GL_ARRAY_BUFFER, pos_buf);
	glBufferData(GL_ARRAY_BUFFER, bufferSIze, init_pos, GL_DYNAMIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);


	glGenBuffers(1, &vel_buf);
	glBindBuffer(GL_ARRAY_BUFFER, vel_buf);
	glBufferData(GL_ARRAY_BUFFER, bufferSIze, init_vel, GL_DYNAMIC_COPY);

	//glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	//glEnableVertexAttribArray(0);


	cudaGraphicsGLRegisterBuffer(&cuda_pos_resource, pos_buf, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&cuda_vel_resource, vel_buf, cudaGraphicsMapFlagsWriteDiscard);


	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);

}


void drawStars() {
	glPointSize(3.0f);
	// 정점의 size 조절 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glColor4f(0.2, 0.42, 0.1, 0.5);

	glBindVertexArray(pos_buf);
	glDrawArrays(GL_POINTS, 0, bufferSIze);
	glBindVertexArray(0);

	//glBegin(GL_POINTS); // mode 선택 
	////glVertex2f(-0.5f, 0.5f);
	////glVertex2f(0.5f, 0.5f);
	////glVertex2f(-0.5f, -0.5f);
	////glVertex2f(0.5f, -0.5f);
	////glVertex2f(0.0f, 0.0f);
	////glVertex2f(1.0f, 1.0f);

	//for (int i = 0; i < star_size; i++) {
	//	//float p = selectPos(), q = selectPos();
	//	/*if (p == 0.0f || q == 0.0f) {
	//		glVertex2f(0, 0);
	//	}
	//	else*/
	//	//glVertex2f(p, q);
	//	POS pos = selectPos();
	//	glVertex2f(pos.x, pos.y);

	//}

	//glEnd();
	//glutSwapBuffers();
	//glFlush();
	glutPostRedisplay();
}

void setup(int W, int H, int screenW, int screenH) {
	POS *pos = 0, *vel = 0;
	cudaGraphicsMapResources(1, &cuda_pos_resource, 0);
	cudaGraphicsMapResources(1, &cuda_vel_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&pos, NULL, cuda_pos_resource);
	cudaGraphicsResourceGetMappedPointer((void **)&vel, NULL, cuda_vel_resource);

	kernelLauncher_star(pos, vel, W, H, screenW, screenH);//cuda 커널 호출

	cudaGraphicsUnmapResources(1, &cuda_pos_resource, 0);
	cudaGraphicsUnmapResources(1, &cuda_vel_resource, 0);
}