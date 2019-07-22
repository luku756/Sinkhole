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
struct cudaGraphicsResource *cuda_pbo_resource;	//openGL�� pixel buffer�� �����ϱ� ���� ����.


//cuda�� �����Ͽ� �׸��� ���� �ؽ��� ����.
void render() {
	uchar4 *d_out = 0;
	cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL,
		cuda_pbo_resource);
	kernelLauncher(d_out, W, H, loc);//cuda Ŀ�� ȣ��
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

//�ؽ��ĸ� ���� �簢�� �׸���
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

//�׸��� �Լ�
void display() {
	//render();
	drawTexture();
	glutSwapBuffers();
}

//GLUT �ʱ�ȭ
void initGLUT(int *argc, char **argv) {
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(screenW, screenH);
	glutCreateWindow(TITLE_STRING);
#ifndef __APPLE__
	glewInit();
#endif
}

//�ȼ� ���� ������Ʈ ����, �ؽ��� ������Ʈ ����. ���� ����. ������ ���۸� cuda�� ����.
void initPixelBuffer() {
	//�ȼ� ���� ������Ʈ ����
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	//�����Ͱ� NULL(0), ������ ���� ������ �Ҵ�.
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * W*H * sizeof(GLubyte), 0, GL_STREAM_DRAW);

	//�ؽ��� ������Ʈ ����
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	//������ pbo�� cuda �� ����.
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}


//���� �ݹ��Լ�(������)
void exitfunc() {
	if (pbo) {
		cudaGraphicsUnregisterResource(cuda_pbo_resource);//�ȼ� ���� ������Ʈ ��� ����
		glDeleteBuffers(1, &pbo);//���� ����
		glDeleteTextures(1, &tex);//���� ����
	}
}

int main(int argc, char** argv) {
	printInstructions();					//how to use
	initGLUT(&argc, argv);					//�ʱ�ȭ
	gluOrtho2D(0, screenW, screenH, 0);					//����. 2D
	glutKeyboardFunc(keyboard);				//Ű���� �Է� �ݹ��Լ�
	glutSpecialFunc(handleSpecialKeypress);	//����� Ű(����Ű ��) �� �Է��� �޴� �ݹ��Լ�
	glutPassiveMotionFunc(mouseMove);		//���콺 �Է� �ݹ��Լ�
	glutMotionFunc(mouseDrag);				//���콺 Ŭ�� �ݹ��Լ�
	glutDisplayFunc(display);				//display �ݹ��Լ�
	initPixelBuffer();						//�ȼ� ����, �ؽ��� ������ cuda ����.
	initialStar(W*H, screenW, screenH);
	glutMainLoop();							//��ο� ����
	atexit(exitfunc);						//���� �ݹ��Լ�
	return 0;
}