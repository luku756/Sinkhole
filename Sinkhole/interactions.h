#ifndef INTERACTIONS_H
#define INTERACTIONS_H
#define W 1000
#define H 1000
#define screenW 1024
#define screenH 1024
#define DELTA 5 // 화살표 키 눌렀을 때 좌표의 변화량.
#define TITLE_STRING "flashlight: distance image display app"	//창 이름
int2 loc = { W / 2, H / 2 };	//광원의 기준점. 초기는 화면의 중점
bool dragMode = false; // 마우스 모드.

//키보드 이벤트 콜백함수. a, esc 입력받음.
void keyboard(unsigned char key, int x, int y) {
	if (key == 'a') dragMode = !dragMode; //마우스 추적 여부 토글
	if (key == 27) exit(0);	//ESC, 종료
	glutPostRedisplay();
}

//마우스 이동 이벤트 콜백함수. 마우스의 현재 위치로 loc을 이동.
void mouseMove(int x, int y) {
	if (dragMode) return;
	loc.x = x;
	loc.y = y;
	glutPostRedisplay();
}

//마우스 클릭하고 움직일 경우 움직이는 중에는 반영되지 않지만, 뗐을 때 그 위치로 이동.
void mouseDrag(int x, int y) {
	if (!dragMode) return;
	loc.x = x;
	loc.y = y;
	glutPostRedisplay();
}

//특수 키 콜백함수. 방향키에 따라 광원의 중점 이동.
void handleSpecialKeypress(int key, int x, int y) {
	if (key == GLUT_KEY_LEFT)  loc.x -= DELTA;
	if (key == GLUT_KEY_RIGHT) loc.x += DELTA;
	if (key == GLUT_KEY_UP)    loc.y -= DELTA;
	if (key == GLUT_KEY_DOWN)  loc.y += DELTA;
	glutPostRedisplay();
}

//초기 안내(how to use)
void printInstructions() {
	printf("flashlight interactions\n");
	printf("a: toggle mouse tracking mode\n");
	printf("arrow keys: move ref location\n");
	printf("esc: close graphics window\n");
}

#endif