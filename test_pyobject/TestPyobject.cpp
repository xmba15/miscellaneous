/**
 * @file    TestPyobject.cpp
 *
 * @brief   Test Pyobject
 *
 * @author  ba tran
 *
 * @date    2019-03-16
 *
 * Copyright (c) organization
 *
 */

#include <Python.h>
#include <iostream>

#if PY_MAJOR_VERSION >= 3
#define PY3K
#endif

class DummyClass
{
public:
  void hello()
  {
    std::cout << "Test pyobject"
              << "\n";
  }
};

static void freePyDummyClass(PyObject *obj)
{
  DummyClass *dc =
      reinterpret_cast<DummyClass *>(PyCapsule_GetPointer(obj, "_DummyClass"));
  delete dc;
}

static PyObject *allocPyDummyClass(PyObject *self, PyObject *args)
{
  DummyClass *const dc = new DummyClass();
  return PyCapsule_New(dc, "_DummyClass", freePyDummyClass);
}

PyObject *pyHello(PyObject *self, PyObject *args)
{
  PyObject *pyObj;
  if (!PyArg_ParseTuple(args, "O", &pyObj)) {
    return nullptr;
  }

  DummyClass *dc = reinterpret_cast<DummyClass *>(
      PyCapsule_GetPointer(pyObj, "_DummyClass"));
  if (dc == nullptr) {
    return nullptr;
  }

  dc->hello();

  Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"dummyclass_alloc", allocPyDummyClass, METH_VARARGS,
     "allocate a Dummy Object"},
    {"hello", pyHello, METH_VARARGS, "call hello() method"},
    {NULL, NULL, 0, NULL}};

#ifdef PY3K
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "example",
                                    "Use C/C++ pointer", -1, methods};

PyMODINIT_FUNC PyInit_example(void)
{
  return PyModule_Create(&module);
}

#else
PyMODINIT_FUNC initHello() {
  Py_InitModule3("example", methods, "Use C/C++ pointer");
}

#endif
