#pragma once

struct GLFWwindow;

class App {
public:
    App();

    bool init();
    int run();

private:
    GLFWwindow* m_window;
};
