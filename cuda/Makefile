BUILD=build
CMAKE_ARGS:=$(CMAKE_ARGS)

all:
	@mkdir -p $(BUILD)
	@cd $(BUILD); cmake .. -DCMAKE_BUILD_TYPE=Release $(CMAKE_ARGS) && $(MAKE)
	@echo -e "\n Now do 'make install' to install this package.\n"

clean:
	@rm -rf $(BUILD)
