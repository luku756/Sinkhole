#ifndef INTERACTIONS_H
#define INTERACTIONS_H
#define W 1000
#define H 1000
#define screenW 1024
#define screenH 1024
#define DELTA 5 // ȭ��ǥ Ű ������ �� ��ǥ�� ��ȭ��.
#define TITLE_STRING "flashlight: distance image display app"	//â �̸�
int2 loc = { W / 2, H / 2 };	//������ ������. �ʱ�� ȭ���� ����
bool dragMode = false; // ���콺 ���.

//Ű���� �̺�Ʈ �ݹ��Լ�. a, esc �Է¹���.
void keyboard(unsigned char key, int x, int y) {
	if (key == 'a') dragMode = !dragMode; //���콺 ���� ���� ���
	if (key == 27) exit(0);	//ESC, ����
	glutPostRedisplay();
}

//���콺 �̵� �̺�Ʈ �ݹ��Լ�. ���콺�� ���� ��ġ�� loc�� �̵�.
void mouseMove(int x, int y) {
	if (dragMode) return;
	loc.x = x;
	loc.y = y;
	glutPostRedisplay();
}

//���콺 Ŭ���ϰ� ������ ��� �����̴� �߿��� �ݿ����� ������, ���� �� �� ��ġ�� �̵�.
void mouseDrag(int x, int y) {
	if (!dragMode) return;
	loc.x = x;
	loc.y = y;
	glutPostRedisplay();
}

//Ư�� Ű �ݹ��Լ�. ����Ű�� ���� ������ ���� �̵�.
void handleSpecialKeypress(int key, int x, int y) {
	if (key == GLUT_KEY_LEFT)  loc.x -= DELTA;
	if (key == GLUT_KEY_RIGHT) loc.x += DELTA;
	if (key == GLUT_KEY_UP)    loc.y -= DELTA;
	if (key == GLUT_KEY_DOWN)  loc.y += DELTA;
	glutPostRedisplay();
}

//�ʱ� �ȳ�(how to use)
void printInstructions() {
	printf("flashlight interactions\n");
	printf("a: toggle mouse tracking mode\n");
	printf("arrow keys: move ref location\n");
	printf("esc: close graphics window\n");
}

#endif