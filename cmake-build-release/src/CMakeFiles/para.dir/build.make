# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /tmp/tmp.VJ6IohNGxK

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /tmp/tmp.VJ6IohNGxK/cmake-build-release

# Include any dependencies generated for this target.
include src/CMakeFiles/para.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/para.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/para.dir/flags.make

src/CMakeFiles/para.dir/utiles/parallelStaff.cpp.o: src/CMakeFiles/para.dir/flags.make
src/CMakeFiles/para.dir/utiles/parallelStaff.cpp.o: ../src/utiles/parallelStaff.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/tmp/tmp.VJ6IohNGxK/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/para.dir/utiles/parallelStaff.cpp.o"
	cd /tmp/tmp.VJ6IohNGxK/cmake-build-release/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/para.dir/utiles/parallelStaff.cpp.o -c /tmp/tmp.VJ6IohNGxK/src/utiles/parallelStaff.cpp

src/CMakeFiles/para.dir/utiles/parallelStaff.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/para.dir/utiles/parallelStaff.cpp.i"
	cd /tmp/tmp.VJ6IohNGxK/cmake-build-release/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /tmp/tmp.VJ6IohNGxK/src/utiles/parallelStaff.cpp > CMakeFiles/para.dir/utiles/parallelStaff.cpp.i

src/CMakeFiles/para.dir/utiles/parallelStaff.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/para.dir/utiles/parallelStaff.cpp.s"
	cd /tmp/tmp.VJ6IohNGxK/cmake-build-release/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /tmp/tmp.VJ6IohNGxK/src/utiles/parallelStaff.cpp -o CMakeFiles/para.dir/utiles/parallelStaff.cpp.s

src/CMakeFiles/para.dir/utiles/parallelStaff.cpp.o.requires:

.PHONY : src/CMakeFiles/para.dir/utiles/parallelStaff.cpp.o.requires

src/CMakeFiles/para.dir/utiles/parallelStaff.cpp.o.provides: src/CMakeFiles/para.dir/utiles/parallelStaff.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/para.dir/build.make src/CMakeFiles/para.dir/utiles/parallelStaff.cpp.o.provides.build
.PHONY : src/CMakeFiles/para.dir/utiles/parallelStaff.cpp.o.provides

src/CMakeFiles/para.dir/utiles/parallelStaff.cpp.o.provides.build: src/CMakeFiles/para.dir/utiles/parallelStaff.cpp.o


# Object files for target para
para_OBJECTS = \
"CMakeFiles/para.dir/utiles/parallelStaff.cpp.o"

# External object files for target para
para_EXTERNAL_OBJECTS =

src/libpara.a: src/CMakeFiles/para.dir/utiles/parallelStaff.cpp.o
src/libpara.a: src/CMakeFiles/para.dir/build.make
src/libpara.a: src/CMakeFiles/para.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/tmp/tmp.VJ6IohNGxK/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libpara.a"
	cd /tmp/tmp.VJ6IohNGxK/cmake-build-release/src && $(CMAKE_COMMAND) -P CMakeFiles/para.dir/cmake_clean_target.cmake
	cd /tmp/tmp.VJ6IohNGxK/cmake-build-release/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/para.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/para.dir/build: src/libpara.a

.PHONY : src/CMakeFiles/para.dir/build

src/CMakeFiles/para.dir/requires: src/CMakeFiles/para.dir/utiles/parallelStaff.cpp.o.requires

.PHONY : src/CMakeFiles/para.dir/requires

src/CMakeFiles/para.dir/clean:
	cd /tmp/tmp.VJ6IohNGxK/cmake-build-release/src && $(CMAKE_COMMAND) -P CMakeFiles/para.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/para.dir/clean

src/CMakeFiles/para.dir/depend:
	cd /tmp/tmp.VJ6IohNGxK/cmake-build-release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /tmp/tmp.VJ6IohNGxK /tmp/tmp.VJ6IohNGxK/src /tmp/tmp.VJ6IohNGxK/cmake-build-release /tmp/tmp.VJ6IohNGxK/cmake-build-release/src /tmp/tmp.VJ6IohNGxK/cmake-build-release/src/CMakeFiles/para.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/para.dir/depend
