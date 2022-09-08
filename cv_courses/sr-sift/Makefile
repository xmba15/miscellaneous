BUILD_TYPE=Release
CMAKE_ARGS:=$(CMAKE_ARGS)

default:
	@mkdir -p build
	@cd build && cmake .. -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
                              -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
                              $(CMAKE_ARGS)
	@cd build && make
