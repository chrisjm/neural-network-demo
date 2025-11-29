#pragma once

#include <string>
#include <optional>

struct GLFWwindow;

// GLFW will call this when something goes wrong at the windowing/OS level
void glfw_error_callback(int error, const char* description);

// Helper to dump OpenGL errors with a label so you can see *where* they came from
void check_gl_error(const char* label);

// Helper function to resize the viewport if the user resizes the window
void framebuffer_size_callback(GLFWwindow* window, int width, int height);

// Helper to load a text file into a std::string (used for shader sources)
std::optional<std::string> loadTextFile(const char* path);
