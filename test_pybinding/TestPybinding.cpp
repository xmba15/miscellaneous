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
#include <pybind11/stl.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>

namespace py = pybind11;

struct inty {
  long long_value;
};

void print(inty s)
{
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

void testVec(const cv::Vec3b &v)
{
  uchar a = v[0];
  std::cout << int(a) << "\n";
}

void testPoint(const cv::Point &p)
{
  std::cout << p.x << " " << p.y << "\n";
}

void testMat(const cv::Mat &img)
{
  std::cout << img.size << "\n";
}

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
  m.def("testVec", &testVec, "A function to test cv::Vec");
  m.def("testPoint", &testPoint, "A function to test cv::Point");
  m.def("testMat", &testMat, "A function to test cv::Mat");
}

namespace pybind11
{
namespace detail
{
template <> struct type_caster<cv::Vec3b> {
public:
  PYBIND11_TYPE_CASTER(cv::Vec3b, _("cv::Vec3b"));

  bool load(py::handle src, bool)
  {
    PyObject *source = src.ptr();

    py::list l = py::reinterpret_borrow<py::list>(source);

    value[0] = l[0].cast<uchar>();
    value[1] = l[1].cast<uchar>();
    value[2] = l[2].cast<uchar>();

    return true;
  }

  static py::handle cast(const cv::Vec3b &vec, py::return_value_policy,
                         py::handle)
  {
    py::list l;
    l.append(vec[0]);
    l.append(vec[1]);
    l.append(vec[2]);
    return l.release();
  }
};

template <class T> void insertValue(cv::Point &p, py::buffer_info &info)
{
  std::vector<T> v(static_cast<T *>(info.ptr),
                   static_cast<T *>(info.ptr) + info.shape[0]);
  p.x = v[0];
  p.y = v[1];
}

template <> struct type_caster<cv::Point> {
public:
  PYBIND11_TYPE_CASTER(cv::Point, _("numpy.ndarray"));

  bool load(py::handle src, bool)
  {
    if (!src) {
      return false;
    }

    if (!py::isinstance<py::array>(src)) {
      throw std::runtime_error(
          "Incompatible type of Point intput. Use numpy array instead");
    }

    py::array parray = reinterpret_borrow<py::array>(src);
    py::buffer_info info = parray.request();

    if (info.ndim != 1) {
      throw std::runtime_error("Number of dimension must be one");
    }

    if (info.shape[0] != 2) {
      throw std::runtime_error("Input shapes must be 2");
    }

    if (info.format == py::format_descriptor<double>::format()) {
      insertValue<double>(value, info);
    } else if (info.format == py::format_descriptor<float>::format()) {
      insertValue<float>(value, info);
    } else if (info.format == py::format_descriptor<int16_t>::format()) {
      insertValue<int16_t>(value, info);
    } else if (info.format == py::format_descriptor<int32_t>::format()) {
      insertValue<int32_t>(value, info);
    } else {
      insertValue<int64_t>(value, info);
    }

    return true;
  }

  static py::handle cast(const cv::Point &p, py::return_value_policy,
                         py::handle)
  {
    std::vector<int> v(p.x, p.y);
    py::array_t<int> parray = py::array_t<int>(v.size());
    py::buffer_info info = parray.request();
    int *infoPtr = static_cast<int *>(info.ptr);
    std::memcpy(infoPtr, v.data(), v.size() * sizeof(int));

    return parray.release();
  }
};

template <> struct type_caster<cv::Mat> {
public:
  PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

  bool load(handle src, bool)
  {
    if (!src) {
      return false;
    }

    if (!py::isinstance<py::array>(src)) {
      throw std::runtime_error(
          "Incompatible type of Mat intput. Use numpy array instead");
    }

    py::array b = py::reinterpret_borrow<py::array>(src);
    buffer_info info = b.request();

    int ndims = info.ndim;

    decltype(CV_32F) dtype;
    size_t elemsize;
    if (info.format == py::format_descriptor<float>::format()) {
      if (ndims == 3) {
        dtype = CV_32FC3;
      } else {
        dtype = CV_32FC1;
      }
      elemsize = sizeof(float);
    } else if (info.format == py::format_descriptor<double>::format()) {
      if (ndims == 3) {
        dtype = CV_64FC3;
      } else {
        dtype = CV_64FC1;
      }
      elemsize = sizeof(double);
    } else if (info.format == py::format_descriptor<unsigned char>::format()) {
      if (ndims == 3) {
        dtype = CV_8UC3;
      } else {
        dtype = CV_8UC1;
      }
      elemsize = sizeof(unsigned char);
    } else {
      throw std::runtime_error("Unsupported type");
    }

    std::vector<int> shape = {static_cast<int>(info.shape[0]),
                              static_cast<int>(info.shape[1])};

    value = cv::Mat(cv::Size(shape[1], shape[0]), dtype, info.ptr,
                    cv::Mat::AUTO_STEP);
    return true;
  }

  static handle cast(const cv::Mat &m, return_value_policy, handle defval)
  {
    std::string format = py::format_descriptor<unsigned char>::format();
    size_t elemsize = sizeof(unsigned char);
    int dim;
    switch (m.type()) {
      case CV_8U:
        format = py::format_descriptor<unsigned char>::format();
        elemsize = sizeof(unsigned char);
        dim = 2;
        break;
      case CV_8UC3:
        format = py::format_descriptor<unsigned char>::format();
        elemsize = sizeof(unsigned char);
        dim = 3;
        break;
      case CV_32F:
        format = py::format_descriptor<float>::format();
        elemsize = sizeof(float);
        dim = 2;
        break;
      case CV_64F:
        format = py::format_descriptor<double>::format();
        elemsize = sizeof(double);
        dim = 2;
        break;
      default:
        throw std::runtime_error("Unsupported type");
    }

    std::vector<size_t> bufferdim;
    std::vector<size_t> strides;
    if (dim == 2) {
      bufferdim = {(size_t)m.rows, (size_t)m.cols};
      strides = {elemsize * (size_t)m.cols, elemsize};
    } else if (dim == 3) {
      bufferdim = {(size_t)m.rows, (size_t)m.cols, (size_t)3};
      strides = {(size_t)elemsize * m.cols * 3, (size_t)elemsize * 3,
                 (size_t)elemsize};
    }
    return array(buffer_info(m.data,   /* Pointer to buffer */
                             elemsize, /* Size of one scalar */
                             format, /* Python struct-style format descriptor */
                             dim,    /* Number of dimensions */
                             bufferdim, /* Buffer dimensions */
                             strides    /* Strides (in bytes) for each index */
                             ))
        .release();
  }
};
}  // namespace detail
}  // namespace pybind11
