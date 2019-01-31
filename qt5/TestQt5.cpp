/**
 * @file    TestQt5
 *
 * @brief   Test Qt5 Visualization
 *
 * @author  xmba15
 *
 * @date    2019-01-31
 *
 * miscellaneous framework
 *
 * Copyright (c) organization
 *
 */

#include <QApplication>
#include <QCoreApplication>
#include <QWidget>
#include <QLabel>
#include <QDebug>
#include <QLinkedList>

int main(int argc, char *argv[]) {
  // QApplication app(argc, argv);
  QCoreApplication app(argc, argv);
  // QWidget window;
  // QLabel *l = new QLabel("hello world");

  QLinkedList<QString> List;
  List << "A" << "B" << "C";
  List.append("D");
  List.append("E");
  List.append("F");

  foreach(QString s, List) qDebug() << s;

  // l->show();

  // window.resize(250, 150);
  // window.setWindowTitle("Simple Example");
  // window.show();

  return app.exec();
}
