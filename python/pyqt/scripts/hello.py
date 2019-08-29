#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from qt import *


def main():
    app = QApplication(sys.argv)
    button = QPushPutton("Hello World", None)
    app.setMainWidget(button)
    button.show()
    app.exec_loop()


if __name__ == '__main__':
    main()
