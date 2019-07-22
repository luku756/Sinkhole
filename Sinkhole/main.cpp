#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "interactions.h"
#include "graphic.h"

// texture and pixel objects
GLuint pbo = 0; // OpenGL pixel buffer object
GLuint tex = 0; // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource;	//openGL의 pixel buffer와 연결하기 위한 변수.


//cuda와 연동하여 그리기 위한 텍스쳐 생성.
void render() {
	uchar4 *d_out = 0;
	cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL,
		cuda_pbo_resource);
	kernelLauncher(d_out, W, H, loc);//cuda 커널 호출
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

//텍스쳐를 입힌 사각형 그리기
void drawTexture() {
	setup(W,H,screenW, screenH);
	drawStars();

	/*glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0, H/2);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(W/2, H/2);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(W/2, 0);
	glEnd();
	glDisable(GL_TEXTURE_2D);*/

}

//그리기 함수
void display() {
	//render();
	drawTexture();
	glutSwapBuffers();
}

//GLUT 초기화
void initGLUT(int *argc, char **argv) {
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(screenW, screenH);
	glutCreateWindow(TITLE_STRING);
#ifndef __APPLE__
	glewInit();
#endif
}

//픽셀 버퍼 오브젝트 생성, 텍스쳐 오브젝트 생성. 둘을 연결. 생성한 버퍼를 cuda와 연결.
void initPixelBuffer() {
	//픽셀 버퍼 오브젝트 생성
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	//데이터가 NULL(0), 데이터 없이 공간만 할당.
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * W*H * sizeof(GLubyte), 0, GL_STREAM_DRAW);

	//텍스쳐 오브젝트 생성
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	//생성한 pbo를 cuda 와 연결.
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}


//종료 콜백함수(마무리)
void exitfunc() {
	if (pbo) {
		cudaGraphicsUnregisterResource(cuda_pbo_resource);//픽셀 버퍼 오브젝트 등록 해제
		glDeleteBuffers(1, &pbo);//버퍼 해제
		glDeleteTextures(1, &tex);//버퍼 해제
	}
}

int main(int argc, char** argv) {
	printInstructions();					//how to use
	initGLUT(&argc, argv);					//초기화
	gluOrtho2D(0, screenW, screenH, 0);					//직교. 2D
	glutKeyboardFunc(keyboard);				//키보드 입력 콜백함수
	glutSpecialFunc(handleSpecialKeypress);	//스페셜 키(방향키 등) 의 입력을 받는 콜백함수
	glutPassiveMotionFunc(mouseMove);		//마우스 입력 콜백함수
	glutMotionFunc(mouseDrag);				//마우스 클릭 콜백함수
	glutDisplayFunc(display);				//display 콜백함수
	initPixelBuffer();						//픽셀 버퍼, 텍스쳐 생성과 cuda 연결.
	initialStar(W*H, screenW, screenH);
	glutMainLoop();							//드로우 시작
	atexit(exitfunc);						//종료 콜백함수
	return 0;
}