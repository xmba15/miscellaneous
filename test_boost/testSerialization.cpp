/**
 * @file    testSerialization.cpp
 *
 * @brief   test boost serialization
 *
 * @author  xmba15
 *
 * @date    2019-01-28
 *
 * miscellaneous framework
 *
 * Copyright (c) organization
 *
 */

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/serialization.hpp>
#include <cassert>
#include <fstream>
#include <string>
#include <iostream>

class MyClass {
public:
  MyClass() : data1(0),  data2(1.0) {}
  ~MyClass() = default;

  int getData1() const { return data1; }
  void setData1(const int data1) { this->data1 = data1; }

  double getData2() const { return data2; }
  void setData2(const double data2) { this->data2 = data2; }

private:
  friend class boost::serialization::access;
  template <class Archive> void serialize(Archive &ar, const unsigned int) {
    ar &BOOST_SERIALIZATION_NVP(data1);
    ar &BOOST_SERIALIZATION_NVP(data2);
  }
  int data1;
  double data2;
};

int main(int argc, char *argv[]) {
  const std::string xmlOutputPath = "output.xml";
  const std::string textOutputPath = "output";

  {
    MyClass m;
    m.setData1(123);
    m.setData2(140.3);
    std::ofstream ofs(xmlOutputPath);
    assert(ofs);
    boost::archive::xml_oarchive oa(ofs);

    std::ofstream tofs(textOutputPath);
    assert(tofs);
    boost::archive::text_oarchive toa(tofs);

    oa << BOOST_SERIALIZATION_NVP(m);
    toa << BOOST_SERIALIZATION_NVP(m);
  }
  {
    std::ifstream ifs(xmlOutputPath);
    assert(ifs);
    boost::archive::xml_iarchive ia(ifs);

    std::ifstream tifs(textOutputPath);
    assert(tifs);
    boost::archive::text_iarchive tia(tifs);

    MyClass m;
    // ia >> BOOST_SERIALIZATION_NVP(m);
    tia >>  BOOST_SERIALIZATION_NVP(m);

    assert(m.getData1() == 123);
    assert(m.getData2() == 140.3);
  }

  return 0;
}
