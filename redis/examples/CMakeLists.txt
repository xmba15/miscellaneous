include(cpp_redis)
__fetch_cpp_redis(
  ${PROJECT_SOURCE_DIR}/cmake
  ${PROJECT_BINARY_DIR}/cpp_redis_for_${PROJECT_NAME}
)

function(__add_example example_name)
  add_executable(${example_name}
    ${PROJECT_SOURCE_DIR}/examples/${example_name}.cpp
  )

  target_include_directories(${example_name}
    PRIVATE
      ${CPP_REDIS_INCLUDES}
  )
  target_compile_features(${example_name}
    PRIVATE
      cxx_std_11
  )

  target_link_libraries(
    ${example_name}
      PUBLIC
        cpp_redis
  )
endfunction(__add_example example_name)

__add_example("TestClient")
