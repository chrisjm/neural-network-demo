cmake_minimum_required(VERSION 3.10)
project(NeuralNetDemo)

set(CMAKE_CXX_STANDARD 17)

# Control whether ImGui-based debug/control UI is compiled in.
# For WebAssembly builds we default this OFF so the WASM build can be a
# clean GPU visualization driven via the exported C API.
if (EMSCRIPTEN)
    option(NNDEMO_ENABLE_IMGUI "Enable ImGui UI for the demo" OFF)
else()
    option(NNDEMO_ENABLE_IMGUI "Enable ImGui UI for the demo" ON)
endif()

if (EMSCRIPTEN)
    # Emscripten provides GLFW/WebGL, so we do not use find_package here.
else()
    find_package(glfw3 3.3 REQUIRED)
    find_package(OpenGL REQUIRED)
endif()

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

if (EMSCRIPTEN)
    # Build for WebAssembly via Emscripten.
    set(CMAKE_EXECUTABLE_SUFFIX ".html")

    add_executable(NeuralNetDemo
        main.cpp
        src/core/App.cpp
        src/core/Scene.cpp
        src/core/WasmApi.cpp
        src/core/DatasetGenerator.cpp
        src/core/FieldVisualizer.cpp
        src/core/PlotGeometry.cpp
        src/core/Trainer.cpp
        src/core/Optimizer.cpp
        src/core/NetworkVisualizer.cpp
        src/core/ControlPanel.cpp
        src/core/GeometryUtils.cpp
        src/core/Input.cpp
        src/core/ToyNet.cpp
        src/render/GLUtils.cpp
        src/render/ShaderProgram.cpp
        src/render/TriangleMesh.cpp
        extern/imgui/imgui.cpp
        extern/imgui/imgui_draw.cpp
        extern/imgui/imgui_widgets.cpp
        extern/imgui/imgui_tables.cpp
        extern/imgui/imgui_demo.cpp
        extern/imgui/backends/imgui_impl_glfw.cpp
        extern/imgui/backends/imgui_impl_opengl3.cpp
    )

    if (NNDEMO_ENABLE_IMGUI)
        target_compile_definitions(NeuralNetDemo PRIVATE NNDEMO_ENABLE_IMGUI)
    endif()

    target_compile_definitions(NeuralNetDemo PRIVATE IMGUI_IMPL_OPENGL_ES3)

    target_link_options(NeuralNetDemo PRIVATE
        "-sUSE_GLFW=3"
        "-sUSE_WEBGL2=1"
        "-sFULL_ES3=1"
        "--preload-file=${CMAKE_SOURCE_DIR}/shaders@/shaders"
        "-sMODULARIZE=1"
        "-sEXPORT_ES6=1"
        "-sENVIRONMENT=web"
        "-sNO_EXIT_RUNTIME=1"
        "-sEXPORTED_FUNCTIONS=['_main','_nn_set_point_size','_nn_set_dataset','_nn_set_auto_train','_nn_step_train','_nn_get_last_loss','_nn_get_last_accuracy','_nn_get_step_count','_nn_get_learning_rate','_nn_get_batch_size','_nn_get_auto_train','_nn_get_dataset_index','_nn_get_num_points','_nn_get_spread','_nn_get_point_size','_nn_set_learning_rate','_nn_set_batch_size','_nn_set_auto_max_epochs','_nn_set_auto_target_loss','_nn_set_use_target_loss_stop','_nn_set_optimizer','_nn_set_momentum','_nn_set_adam_beta1','_nn_set_adam_beta2','_nn_set_adam_eps','_nn_set_init_mode','_nn_set_probe_enabled','_nn_set_probe_position','_nn_get_auto_max_epochs','_nn_get_auto_target_loss','_nn_get_use_target_loss_stop','_nn_get_optimizer','_nn_get_momentum','_nn_get_adam_beta1','_nn_get_adam_beta2','_nn_get_adam_eps','_nn_get_init_mode','_nn_get_probe_enabled','_nn_get_probe_x','_nn_get_probe_y','_nn_get_selected_point_index','_nn_get_selected_label','_nn_get_max_points']"
    )
else()
    add_definitions(-DIMGUI_IMPL_OPENGL_LOADER_GLAD)

    add_executable(NeuralNetDemo
        main.cpp
        src/core/App.cpp
        src/core/Scene.cpp
        src/core/DatasetGenerator.cpp
        src/core/FieldVisualizer.cpp
        src/core/PlotGeometry.cpp
        src/core/Trainer.cpp
        src/core/Optimizer.cpp
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

    if (NNDEMO_ENABLE_IMGUI)
        target_compile_definitions(NeuralNetDemo PRIVATE NNDEMO_ENABLE_IMGUI)
    endif()

    # Link the libraries for native build
    target_link_libraries(NeuralNetDemo glfw OpenGL::GL)
endif()
