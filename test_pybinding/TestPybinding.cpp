/**
 * @file    TestPybinding.cpp
 *
 * @brief   Test pybind11
 *
 * @author  btran
 *
 * @date    2019-03-11
 *
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <sstream>
#include <string>

struct inty { long long_value; };

void print(inty s) {
    std::cout << s.long_value << std::endl;
}

int add(int i, int j)
{
  return i + j;
}

void testRect(const cv::Rect &r)
{
  std::stringstream ss;
  ss << r.x << " " << r.y << " " << r.width << " " << r.height << " ";
  std::cout << ss.str() << "\n";
}

namespace py = pybind11;

PYBIND11_MODULE(example, m)
{
  m.doc() = "pybind11 example plugin";

  m.def("add", &add, "A function which adds two numbers");

  py::class_<cv::Rect>(m, "Rectangle")
      .def("__repr__",
           [](cv::Rect &r) {
             std::stringstream ss;
             ss << r.x << " " << r.y << " " << r.width << " " << r.height
                << " ";
             return ss.str();
           })
      .def(py::init<>())
      .def(py::init<float, float, float, float>())
      .def_readwrite("x", &cv::Rect::x)
      .def_readwrite("y", &cv::Rect::y)
      .def_readwrite("width", &cv::Rect::width)
      .def_readwrite("height", &cv::Rect::height);

  m.def("testRect", &testRect, "A function to test cv::Rect");
  m.def("print", &print, "A function to test cv::Rect");
}

// namespace pybind11
// {
// namespace detail
// {
// template <> struct type_caster<cv::Point> {
//   PYBIND11_TYPE_CASTER(cv::Point, _("numpy.ndarray"));

//   // Cast numpy to cv::Point
//   bool load(handle src, bool imp)
//   {
    
//     return true;
//   }

//   // Cast cv::Point to numpy
//   // static handle cast(const cv::Point &p, return_value_policy, handle defval)
//   // {
//   // }
// }
// }  // namespace detail
// }  // namespace pybind11

namespace pybind11 { namespace detail {
    template <> struct type_caster<inty> {
    public:
        /**
         * This macro establishes the name 'inty' in
         * function signatures and declares a local variable
         * 'value' of type inty
         */
        PYBIND11_TYPE_CASTER(inty, _("inty"));

        /**
         * Conversion part 1 (Python->C++): convert a PyObject into a inty
         * instance or return false upon failure. The second argument
         * indicates whether implicit conversions should be applied.
         */
        bool load(handle src, bool) {
            /* Extract PyObject from handle */
            PyObject *source = src.ptr();
            /* Try converting into a Python integer value */
            PyObject *tmp = PyNumber_Long(source);
            if (!tmp)
                return false;
            /* Now try to convert into a C++ int */
            value.long_value = PyLong_AsLong(tmp);
            Py_DECREF(tmp);
            /* Ensure return code was OK (to avoid out-of-range errors etc) */
            return !(value.long_value == -1 && !PyErr_Occurred());
        }

        /**
         * Conversion part 2 (C++ -> Python): convert an inty instance into
         * a Python object. The second and third arguments are used to
         * indicate the return value policy and parent object (for
         * ``return_value_policy::reference_internal``) and are generally
         * ignored by implicit casters.
         */
        static handle cast(inty src, return_value_policy /* policy */, handle /* parent */) {
            return PyLong_FromLong(src.long_value);
        }
    };
}} // namespace pybind11::detail
