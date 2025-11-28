cmake_minimum_required(VERSION 3.10)
project(HelloTriangle)

set(CMAKE_CXX_STANDARD 17)

find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)

# If you use a package manager like Vcpkg or Conan, they handle this.
# For manual setups, you might need to include glad.c if not linking a library.
# Here we assume you have the glad library available.

add_executable(HelloTriangle main.cpp)

# Link the libraries
target_link_libraries(HelloTriangle glfw OpenGL::GL)
