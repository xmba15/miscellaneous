/**
 * @file    HelloWorld.cpp
 *
 * @author  btran
 *
 * @date    2019-09-07
 *
 * Copyright (c) organization
 *
 */

#include <GL/freeglut.h>

void drawScene(void)
{
    glClear(GL_COLOR_BUFFER_BIT);

    glColor3f(1.0, 0.0, 0.0);

    // Draw a polygon with specified vertices.
    glBegin(GL_POLYGON);
       glColor3f(1.0, 0.0, 0.0);
       glVertex3f(20.0, 20.0, 0.0);
       glColor3f(0.0, 1.0, 0.0);
       glVertex3f(80.0, 20.0, 0.0);
       glColor3f(0.0, 0.0, 1.0);
       glVertex3f(80.0, 80.0, 0.0);
       glColor3f(1.0, 1.0, 0.0);
       glVertex3f(20.0, 80.0, 0.0);
    glEnd();

    glFlush();
}

// Initialization routine.
void setup(void)
{
    glClearColor(1.0, 1.0, 1.0, 0.0);
}

// OpenGL window reshape routine.
void resize(int w, int h)
{
    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 100.0, 0.0, 100.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

// Keyboard input processing routine.
void keyInput(unsigned char key, int x, int y)
{
    switch (key) {
        case 27:
            exit(0);
            break;
        default:
            break;
    }
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);

    glutInitContextVersion(4, 3);
    glutInitContextProfile(GLUT_COMPATIBILITY_PROFILE);

    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(0, 0);

    glutCreateWindow("Hello World");

    glutDisplayFunc(drawScene);
    glutReshapeFunc(resize);
    glutKeyboardFunc(keyInput);

    setup();

    glutMainLoop();
}
