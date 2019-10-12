cmake_minimum_required(VERSION 3.8)

project(cpp_redis-download NONE)

include(ExternalProject)
ExternalProject_Add(
  cpp_redis
  SOURCE_DIR "@CPP_REDIS_DOWNLOAD_ROOT@/cpp_redis-src"
  BINARY_DIR "@CPP_REDIS_DOWNLOAD_ROOT@/cpp_redis-build"
  GIT_REPOSITORY
    https://github.com/cpp-redis/cpp_redis.git
  GIT_TAG
    4.3.1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
