 #include "App.h"

int main() {
    App app;
    if (!app.init()) {
        return -1;
    }
    return app.run();
}
