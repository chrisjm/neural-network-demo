cmake_minimum_required(VERSION 3.10)
project(NeuralNetDemo)

set(CMAKE_CXX_STANDARD 17)

find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)

# If you use a package manager like Vcpkg or Conan, they handle this.
# For manual setups, you might need to include glad.c if not linking a library.
# Here we assume you have the glad library available.

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/core
    ${CMAKE_SOURCE_DIR}/include/render
    ${CMAKE_SOURCE_DIR}/extern/glad/include
    ${CMAKE_SOURCE_DIR}/extern/imgui
    ${CMAKE_SOURCE_DIR}/extern/imgui/backends
)

add_definitions(-DIMGUI_IMPL_OPENGL_LOADER_GLAD)

add_executable(NeuralNetDemo
    main.cpp
    src/core/App.cpp
    src/core/DatasetGenerator.cpp
    src/core/FieldVisualizer.cpp
    src/core/PlotGeometry.cpp
    src/core/Trainer.cpp
    src/core/NetworkVisualizer.cpp
    src/core/ControlPanel.cpp
    src/core/GeometryUtils.cpp
    src/core/Input.cpp
    src/core/ToyNet.cpp
    src/render/GLUtils.cpp
    src/render/ShaderProgram.cpp
    src/render/TriangleMesh.cpp
    extern/glad/src/glad.c
    extern/imgui/imgui.cpp
    extern/imgui/imgui_draw.cpp
    extern/imgui/imgui_widgets.cpp
    extern/imgui/imgui_tables.cpp
    extern/imgui/imgui_demo.cpp
    extern/imgui/backends/imgui_impl_glfw.cpp
    extern/imgui/backends/imgui_impl_opengl3.cpp
)

# Link the libraries
target_link_libraries(NeuralNetDemo glfw OpenGL::GL)
