# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kszmmnn/porr/c-image-filters-in-parallel

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kszmmnn/porr/c-image-filters-in-parallel/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/c-image-filters-in-parallel.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/c-image-filters-in-parallel.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/c-image-filters-in-parallel.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/c-image-filters-in-parallel.dir/flags.make

CMakeFiles/c-image-filters-in-parallel.dir/main.c.o: CMakeFiles/c-image-filters-in-parallel.dir/flags.make
CMakeFiles/c-image-filters-in-parallel.dir/main.c.o: ../main.c
CMakeFiles/c-image-filters-in-parallel.dir/main.c.o: CMakeFiles/c-image-filters-in-parallel.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kszmmnn/porr/c-image-filters-in-parallel/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/c-image-filters-in-parallel.dir/main.c.o"
	/usr/sbin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/c-image-filters-in-parallel.dir/main.c.o -MF CMakeFiles/c-image-filters-in-parallel.dir/main.c.o.d -o CMakeFiles/c-image-filters-in-parallel.dir/main.c.o -c /home/kszmmnn/porr/c-image-filters-in-parallel/main.c

CMakeFiles/c-image-filters-in-parallel.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/c-image-filters-in-parallel.dir/main.c.i"
	/usr/sbin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/kszmmnn/porr/c-image-filters-in-parallel/main.c > CMakeFiles/c-image-filters-in-parallel.dir/main.c.i

CMakeFiles/c-image-filters-in-parallel.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/c-image-filters-in-parallel.dir/main.c.s"
	/usr/sbin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/kszmmnn/porr/c-image-filters-in-parallel/main.c -o CMakeFiles/c-image-filters-in-parallel.dir/main.c.s

CMakeFiles/c-image-filters-in-parallel.dir/benchmarking/benchmark.c.o: CMakeFiles/c-image-filters-in-parallel.dir/flags.make
CMakeFiles/c-image-filters-in-parallel.dir/benchmarking/benchmark.c.o: ../benchmarking/benchmark.c
CMakeFiles/c-image-filters-in-parallel.dir/benchmarking/benchmark.c.o: CMakeFiles/c-image-filters-in-parallel.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kszmmnn/porr/c-image-filters-in-parallel/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/c-image-filters-in-parallel.dir/benchmarking/benchmark.c.o"
	/usr/sbin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/c-image-filters-in-parallel.dir/benchmarking/benchmark.c.o -MF CMakeFiles/c-image-filters-in-parallel.dir/benchmarking/benchmark.c.o.d -o CMakeFiles/c-image-filters-in-parallel.dir/benchmarking/benchmark.c.o -c /home/kszmmnn/porr/c-image-filters-in-parallel/benchmarking/benchmark.c

CMakeFiles/c-image-filters-in-parallel.dir/benchmarking/benchmark.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/c-image-filters-in-parallel.dir/benchmarking/benchmark.c.i"
	/usr/sbin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/kszmmnn/porr/c-image-filters-in-parallel/benchmarking/benchmark.c > CMakeFiles/c-image-filters-in-parallel.dir/benchmarking/benchmark.c.i

CMakeFiles/c-image-filters-in-parallel.dir/benchmarking/benchmark.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/c-image-filters-in-parallel.dir/benchmarking/benchmark.c.s"
	/usr/sbin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/kszmmnn/porr/c-image-filters-in-parallel/benchmarking/benchmark.c -o CMakeFiles/c-image-filters-in-parallel.dir/benchmarking/benchmark.c.s

CMakeFiles/c-image-filters-in-parallel.dir/filters/convolution.c.o: CMakeFiles/c-image-filters-in-parallel.dir/flags.make
CMakeFiles/c-image-filters-in-parallel.dir/filters/convolution.c.o: ../filters/convolution.c
CMakeFiles/c-image-filters-in-parallel.dir/filters/convolution.c.o: CMakeFiles/c-image-filters-in-parallel.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kszmmnn/porr/c-image-filters-in-parallel/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/c-image-filters-in-parallel.dir/filters/convolution.c.o"
	/usr/sbin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/c-image-filters-in-parallel.dir/filters/convolution.c.o -MF CMakeFiles/c-image-filters-in-parallel.dir/filters/convolution.c.o.d -o CMakeFiles/c-image-filters-in-parallel.dir/filters/convolution.c.o -c /home/kszmmnn/porr/c-image-filters-in-parallel/filters/convolution.c

CMakeFiles/c-image-filters-in-parallel.dir/filters/convolution.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/c-image-filters-in-parallel.dir/filters/convolution.c.i"
	/usr/sbin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/kszmmnn/porr/c-image-filters-in-parallel/filters/convolution.c > CMakeFiles/c-image-filters-in-parallel.dir/filters/convolution.c.i

CMakeFiles/c-image-filters-in-parallel.dir/filters/convolution.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/c-image-filters-in-parallel.dir/filters/convolution.c.s"
	/usr/sbin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/kszmmnn/porr/c-image-filters-in-parallel/filters/convolution.c -o CMakeFiles/c-image-filters-in-parallel.dir/filters/convolution.c.s

CMakeFiles/c-image-filters-in-parallel.dir/filters/functional.c.o: CMakeFiles/c-image-filters-in-parallel.dir/flags.make
CMakeFiles/c-image-filters-in-parallel.dir/filters/functional.c.o: ../filters/functional.c
CMakeFiles/c-image-filters-in-parallel.dir/filters/functional.c.o: CMakeFiles/c-image-filters-in-parallel.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kszmmnn/porr/c-image-filters-in-parallel/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/c-image-filters-in-parallel.dir/filters/functional.c.o"
	/usr/sbin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/c-image-filters-in-parallel.dir/filters/functional.c.o -MF CMakeFiles/c-image-filters-in-parallel.dir/filters/functional.c.o.d -o CMakeFiles/c-image-filters-in-parallel.dir/filters/functional.c.o -c /home/kszmmnn/porr/c-image-filters-in-parallel/filters/functional.c

CMakeFiles/c-image-filters-in-parallel.dir/filters/functional.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/c-image-filters-in-parallel.dir/filters/functional.c.i"
	/usr/sbin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/kszmmnn/porr/c-image-filters-in-parallel/filters/functional.c > CMakeFiles/c-image-filters-in-parallel.dir/filters/functional.c.i

CMakeFiles/c-image-filters-in-parallel.dir/filters/functional.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/c-image-filters-in-parallel.dir/filters/functional.c.s"
	/usr/sbin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/kszmmnn/porr/c-image-filters-in-parallel/filters/functional.c -o CMakeFiles/c-image-filters-in-parallel.dir/filters/functional.c.s

# Object files for target c-image-filters-in-parallel
c__image__filters__in__parallel_OBJECTS = \
"CMakeFiles/c-image-filters-in-parallel.dir/main.c.o" \
"CMakeFiles/c-image-filters-in-parallel.dir/benchmarking/benchmark.c.o" \
"CMakeFiles/c-image-filters-in-parallel.dir/filters/convolution.c.o" \
"CMakeFiles/c-image-filters-in-parallel.dir/filters/functional.c.o"

# External object files for target c-image-filters-in-parallel
c__image__filters__in__parallel_EXTERNAL_OBJECTS =

c-image-filters-in-parallel: CMakeFiles/c-image-filters-in-parallel.dir/main.c.o
c-image-filters-in-parallel: CMakeFiles/c-image-filters-in-parallel.dir/benchmarking/benchmark.c.o
c-image-filters-in-parallel: CMakeFiles/c-image-filters-in-parallel.dir/filters/convolution.c.o
c-image-filters-in-parallel: CMakeFiles/c-image-filters-in-parallel.dir/filters/functional.c.o
c-image-filters-in-parallel: CMakeFiles/c-image-filters-in-parallel.dir/build.make
c-image-filters-in-parallel: CMakeFiles/c-image-filters-in-parallel.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kszmmnn/porr/c-image-filters-in-parallel/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking C executable c-image-filters-in-parallel"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/c-image-filters-in-parallel.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/c-image-filters-in-parallel.dir/build: c-image-filters-in-parallel
.PHONY : CMakeFiles/c-image-filters-in-parallel.dir/build

CMakeFiles/c-image-filters-in-parallel.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/c-image-filters-in-parallel.dir/cmake_clean.cmake
.PHONY : CMakeFiles/c-image-filters-in-parallel.dir/clean

CMakeFiles/c-image-filters-in-parallel.dir/depend:
	cd /home/kszmmnn/porr/c-image-filters-in-parallel/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kszmmnn/porr/c-image-filters-in-parallel /home/kszmmnn/porr/c-image-filters-in-parallel /home/kszmmnn/porr/c-image-filters-in-parallel/cmake-build-debug /home/kszmmnn/porr/c-image-filters-in-parallel/cmake-build-debug /home/kszmmnn/porr/c-image-filters-in-parallel/cmake-build-debug/CMakeFiles/c-image-filters-in-parallel.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/c-image-filters-in-parallel.dir/depend

