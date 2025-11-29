cmake_minimum_required(VERSION 3.10)
project(HelloTriangle)

set(CMAKE_CXX_STANDARD 17)

find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)

# If you use a package manager like Vcpkg or Conan, they handle this.
# For manual setups, you might need to include glad.c if not linking a library.
# Here we assume you have the glad library available.

include_directories(
    glad/include
    imgui
    imgui/backends
)

add_definitions(-DIMGUI_IMPL_OPENGL_LOADER_GLAD)

add_executable(HelloTriangle
    main.cpp
    ShaderProgram.cpp
    TriangleMesh.cpp
    App.cpp
    GLUtils.cpp
    Input.cpp
    GeometryUtils.cpp
    glad/src/glad.c
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_widgets.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_demo.cpp
    imgui/backends/imgui_impl_glfw.cpp
    imgui/backends/imgui_impl_opengl3.cpp
)

# Link the libraries
target_link_libraries(HelloTriangle glfw OpenGL::GL)
