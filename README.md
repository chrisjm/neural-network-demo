# Neural Network Visualization using OpenGL

An interactive C++ / OpenGL demo that started as a shader playground and evolved into a **neural network visualizer**.

This project was meant to help me (and others) understand **both**:

- **Neural network math**: forward pass, softmax + cross-entropy, manual backprop.
- **Shader & GPU pipeline**: how data flows from CPU-side C++ into OpenGL shaders and how the same network is evaluated on the GPU for real‑time visualization.

---

## Features

- **Interactive 2D classifier**

  - Small MLP with architecture **2 → 4 → 8 → 2** (2D inputs, 2 classes).
  - Synthetic datasets:
    - `TwoBlobs`
    - `ConcentricCircles`
    - `TwoMoons`
    - `XORQuads`
    - `Spirals`
  - Live training with step/auto modes, plus loss and accuracy plots.

- **GPU decision field via shaders**

  - CPU network implementation in `ToyNet`.
  - The _same_ network is reimplemented in GLSL in `shaders/field.frag`.
  - Weights and biases are uploaded as uniform arrays each frame:
    - `u_W1`, `u_b1`, `u_W2`, `u_b2`, `u_W3`, `u_b3`.
  - The fragment shader evaluates the NN at each pixel and colors the background by predicted class probability.

- **Interactive exploration of the network**

  - Scatter plot of dataset points rendered as OpenGL point sprites.
  - Click near a point to select it and set a **probe** input.
  - Probe drives:
    - Per-sample prediction readout (probabilities + predicted class).
    - A **network diagram** showing nodes and edges:
      - Line color and thickness encode weight sign and magnitude.
      - Node color encodes activation for the probe point.
      - Tooltips show weights, biases, activations, and contributions.

- **Modern OpenGL + Dear ImGui**
  - GLFW + GLAD + OpenGL 3.3 core profile.
  - Dear ImGui for UI panels and plots.
  - Minimal `ShaderProgram` wrapper around shader compilation/linking and uniforms.

---

## Project structure

- **`CMakeLists.txt`** – CMake build configuration (target: `NeuralNetDemo`).
- **`main.cpp`** – entry point, creates and runs `App`.
- **`include/core/`**
  - `App.h` – application class and main render loop interface.
  - `ControlPanel.h` – UI state and ImGui control panel.
  - `DatasetGenerator.h` – synthetic 2D dataset definitions and helpers.
  - `ToyNet.h` – CPU neural network model.
  - `Trainer.h` – training loop, batching, history of loss/accuracy.
  - `FieldVisualizer.h` – geometry and buffers for the decision field.
  - `NetworkVisualizer.h` – ImGui-based network diagram.
  - `Input.h` – keyboard and mouse handling, including probe selection.
  - `Scene.h` – shared scene utilities (frame context, per-frame update, scene init).
  - `WasmScene.h` – wasm-only scene state and declarations for the small C API.
  - `GeometryUtils.h`, `PlotGeometry.h`, `DataPoint.h` – helpers for geometry and data.
- **`include/render/`**
  - `ShaderProgram.h` – simple RAII wrapper for an OpenGL shader program.
  - `GLUtils.h`, `Object2D.h`, `TriangleMesh.h` – OpenGL utilities and geometry.
- **`src/core/`** – implementations of the core components above.
  - `Scene.cpp` – implementation of shared scene helpers and per-frame update.
  - `WasmApi.cpp` – wasm-only implementation of the exported C API used from JS.
- **`src/render/`** – implementations of rendering utilities and `ShaderProgram`.
- **`shaders/`** – GLSL shaders:
  - `point.vert`, `point.frag` – scatter plot point shader.
  - `grid.vert`, `grid.frag` – grid and axes lines.
  - `field.vert`, `field.frag` – decision boundary field mesh + NN fragment shader.
  - `basic.vert`, `basic.frag` – simple test shaders.
- **`extern/`** – vendored third-party code (GLAD, Dear ImGui and backends).

---

## Building

### Prerequisites

- **CMake** ≥ 3.10
- **C++17** compiler (GCC, Clang, or MSVC)
- **OpenGL 3.3+** capable GPU and drivers
- **GLFW 3.3** and OpenGL dev libs available to CMake (e.g. via a package manager)

On macOS or Linux, you can install GLFW and OpenGL headers via your package manager (Homebrew, apt, etc.). On Windows you can use vcpkg, Conan, or manually install libraries, as long as CMake can find `glfw3` and `OpenGL`.

### Configure and build

From the project root:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

This produces an executable named **`NeuralNetDemo`** in the build directory.

### Run

From the `build/` directory:

```bash
./NeuralNetDemo   # or NeuralNetDemo.exe on Windows
```

The app loads shaders from the `shaders/` directory **relative to the current working directory**, so it is easiest to run it from the build directory created above.

---

## WebAssembly build & web integration

The project also builds to WebAssembly using Emscripten. From the project root:

```bash
./build_all.sh
```

This will:

- **Native**: configure and build to `build/` (desktop OpenGL app).
- **WASM**: configure and build to `build-wasm/` via `emcmake`.

The wasm build produces (in `build-wasm/`):

- `NeuralNetDemo.js` – ES module loader (built with `-sMODULARIZE=1 -sEXPORT_ES6=1`).
- `NeuralNetDemo.wasm` – WebAssembly binary.
- `NeuralNetDemo.data` – preloaded shader assets.

### Running the wasm build locally

From the project root:

```bash
./serve_wasm.sh
```

This serves `build-wasm/` at <http://localhost:8000>. Open `NeuralNetDemo.html` in your browser to see the standalone demo.

### Embedding the wasm module in your own app (e.g. SvelteKit)

Because the wasm build is configured as a modular ES6 loader, you can import it from your own frontend instead of using the generated HTML shell.

1. **Copy the build artifacts** somewhere your app can serve them from, for example:

   - `static/nn/NeuralNetDemo.js`
   - `static/nn/NeuralNetDemo.wasm`
   - `static/nn/NeuralNetDemo.data`

2. **Import and instantiate the module** from your app code. In Svelte/SvelteKit (e.g. in a component using `onMount`):

```ts
import { onMount } from "svelte";
import createModule from "/nn/NeuralNetDemo.js"; // adjust path as needed

let module: any;

onMount(async () => {
  const canvas = document.getElementById("demo-canvas") as HTMLCanvasElement;

  module = await createModule({
    canvas,
    print: console.log,
    printErr: console.error,
  });
});
```

The C++ side calls `emscripten_set_main_loop`, so once the module is instantiated, the render loop starts automatically.

### Small C API exposed to JavaScript

The wasm build exposes a tiny C API (via `extern "C"`), which Emscripten exports as functions on the module object:

- `void nn_set_point_size(float size);`
- `void nn_set_dataset(int datasetIndex, int numPoints, float spread);`
- `void nn_set_auto_train(int enabled);`
- `void nn_step_train();`

Read-back functions:

- `float nn_get_last_loss();`
- `float nn_get_last_accuracy();`
- `int   nn_get_step_count();`
- `float nn_get_learning_rate();`
- `int   nn_get_batch_size();`
- `int   nn_get_auto_train();`
- `int   nn_get_dataset_index();`
- `int   nn_get_num_points();`
- `float nn_get_spread();`
- `float nn_get_point_size();`

These are compiled with `-sEXPORTED_FUNCTIONS=['_main', '_nn_set_point_size', '_nn_set_dataset', '_nn_set_auto_train', '_nn_step_train', '_nn_get_last_loss', '_nn_get_last_accuracy', '_nn_get_step_count', '_nn_get_learning_rate', '_nn_get_batch_size', '_nn_get_auto_train', '_nn_get_dataset_index', '_nn_get_num_points', '_nn_get_spread', '_nn_get_point_size']`, so from JS you can call:

```ts
// Example helpers after module = await createModule(...)

function setPointSize(size: number) {
  module._nn_set_point_size(size);
}

function setDataset(datasetIndex: number, numPoints: number, spread: number) {
  module._nn_set_dataset(datasetIndex, numPoints, spread);
}

function setAutoTrain(enabled: boolean) {
  module._nn_set_auto_train(enabled ? 1 : 0);
}

function stepTrainOnce() {
  module._nn_step_train();
}

// Example of polling metrics from Svelte/SvelteKit
function getMetrics() {
  return {
    loss: module._nn_get_last_loss(),
    accuracy: module._nn_get_last_accuracy(),
    step: module._nn_get_step_count(),
    learningRate: module._nn_get_learning_rate(),
    batchSize: module._nn_get_batch_size(),
    autoTrain: !!module._nn_get_auto_train(),
    datasetIndex: module._nn_get_dataset_index(),
    numPoints: module._nn_get_num_points(),
    spread: module._nn_get_spread(),
    pointSize: module._nn_get_point_size(),
  };
}
```

`datasetIndex` maps directly to the `DatasetType` enum in `DatasetGenerator.h`:

- `0` → `TwoBlobs`
- `1` → `ConcentricCircles`
- `2` → `TwoMoons`
- `3` → `XORQuads`
- `4` → `Spirals`

With this minimal C API you can:

- Drive point size and dataset parameters from your own UI (e.g. Svelte controls).
- Toggle auto training from the host app.
- Trigger single training steps on demand.

The rendering loop, ImGui UI, and neural net visualization continue to run entirely inside the wasm module; your app just sends high-level control commands.

---

## How to use the demo

### Main view

- Large central plot shows:
  - The **dataset** as colored points.
  - The **decision field** (background) colored by the network’s predicted class probabilities.

### Neural Net Controls panel

The "Neural Net Controls" ImGui window lets you:

- **Choose a dataset**

  - `Dataset` combo box selects between `TwoBlobs`, `ConcentricCircles`, `TwoMoons`, `XORQuads`, and `Spirals`.
  - `Points` slider controls how many samples are generated.
  - `Spread` slider adjusts noise or radial spread depending on dataset.
  - **Regenerate Data** button regenerates and re-uploads the dataset to the GPU and resets training.

- **Point rendering**

  - `Point Size` slider controls the OpenGL point size.

- **Probe and selection**

  - `Activation Probe Enabled` toggles whether a probe is used.
  - `Probe X` / `Probe Y` sliders set the probe coordinate in clip/NDC space.
  - Clicking near a point in the plot (outside of any ImGui windows):
    - Selects the closest point in a small radius.
    - Updates the probe position to that sample.
    - Marks the point as selected (drawn with an outline) and records its label.
  - When a point is selected, the panel shows:
    - Its index, coordinates, and class label.
    - The predicted probabilities `p0` and `p1` and the predicted class.

--- **Training controls**

- `Learning Rate` slider controls how big each weight update step is.
- `Batch Size` slider controls how many samples are used per training step.
- `Optimizer` combo selects how gradients are turned into weight updates:
  - `SGD` – plain stochastic gradient descent.
  - `SGD + Momentum` – adds a "velocity" term that smooths noisy gradients and helps push through shallow regions.
  - `Adam` – adaptive optimizer that keeps moving averages of gradients and their squares (β1, β2) for per-parameter step sizes.
- When `SGD + Momentum` is selected:
  - `Momentum` controls how strongly the optimizer keeps moving in the previous update direction (0 = no momentum, close to 1 = very smooth but can overshoot).
- When `Adam` is selected:
  - `Adam Beta1` controls how quickly the first-moment (mean gradient) estimate forgets old information.
  - `Adam Beta2` controls how quickly the second-moment (squared-gradient) estimate forgets old information.
  - `Adam Eps` is a small constant added inside the square root to keep divisions numerically stable.
- `Train Epoch` runs a single training step (one minibatch) with the current optimizer.
- `Auto Train` toggles continuous training.
- `Epoch`, `Loss`, and `Accuracy` display the latest training stats ("epoch" here effectively counts training steps).
- `Auto Max Epochs` sets an optional upper bound on auto training steps; `0` disables the epoch-based limit (default is 500).
- `Stop on Target Loss` toggles an optional loss-based stopping rule, which uses `Auto Target Loss` as a threshold; a value of `0.0` disables loss-based stopping.
- **Loss Plot** and **Accuracy Plot** windows track training history over time.

### Network diagram

The "Network Diagram" ImGui window shows a schematic of the MLP:

- Architecture: **Input (2) → Hidden1 (4 ReLU) → Hidden2 (8 ReLU) → Output (2)**.
- Node positions are laid out left to right per layer.
- **Weights**
  - Drawn as lines between nodes.
  - Line color encodes weight sign; thickness encodes |weight|.
  - Hovering edges reveals the exact weight value and (with a probe) the activation contribution.
- **Biases**
  - Nodes with larger |bias| show a halo around them.
- **Activations**
  - With the probe enabled, node color encodes its activation at that probe point, using a warm orange for strongly positive activations and a cool blue for strongly negative/suppressed activations.
  - Hovering nodes shows layer name, index, bias, and activation (or output probabilities).

---

## Neural network math (CPU implementation)

The core neural network is implemented in **`ToyNet`**:

- Architecture constants:
  - `InputDim = 2`, `Hidden1 = 4`, `Hidden2 = 8`, `OutputDim = 2`.
- Parameters:
  - Weight matrices: `W1`, `W2`, `W3` stored as 1D vectors.
  - Bias vectors: `b1`, `b2`, `b3`.
- Forward pass for a batch (`ToyNet::trainBatch`):

  - Layer 1: `z1 = a0 · W1^T + b1`, `a1 = ReLU(z1)`.
  - Layer 2: `z2 = a1 · W2^T + b2`, `a2 = ReLU(z2)`.
  - Output layer: `logits = a2 · W3^T + b3`.
  - Softmax: `p_k = exp(logit_k − max_logit) / Σ_j exp(logit_j − max_logit)`.
  - Loss: cross-entropy over the correct class, averaged over the batch.
  - Accuracy: fraction of samples where `argmax(p)` equals the true label.

-- Backpropagation:

- Uses **softmax + cross-entropy gradient**: `dL/dz3 = p − y`.
- Propagates gradients through each layer, applying ReLU derivative (`1` if pre-activation > 0, else `0`).
- Accumulates gradients for all weights and biases across the batch.
- Averages gradients, then applies an optimizer step (SGD, SGD + Momentum, or Adam) using the current learning rate and optimizer hyperparameters.

- Single-sample forward (`forwardSingle` / `forwardSingleWithActivations`):
  - Runs the same math as above for one `(x, y)` pair.
  - Returns probabilities `p0` and `p1` and (optionally) hidden layer activations.
  - Used by the UI for prediction readouts and by the network diagram when probing.

**`Trainer`** wraps `ToyNet` and adds:

- Configurable `learningRate`, `batchSize`, optimizer type (`SGD`, `SGD + Momentum`, `Adam`), and optimizer hyperparameters (`momentum`, `adamBeta1`, `adamBeta2`, `adamEps`).
- Auto-training controls (`autoTrain`, stopping conditions).
- History buffers for loss and accuracy for plotting.
- Functions to create mini-batches and perform one or many training steps.

Synthetic datasets are created in **`DatasetGenerator`**:

- Each dataset has a different decision boundary (e.g. blobs, circles, moons, spiral),
  providing good intuition about what the NN is trying to learn.

---

## Shader pipeline & GPU data flow

The project is intentionally structured so that you can see how the **same neural network** runs on the GPU using OpenGL shaders.

### CPU side: feeding the network into the GPU

In `App::renderLoop` (see `src/core/App.cpp`):

1. After each training step, the current network parameters are read from `Trainer::net`:
   - `W1`, `B1`, `W2`, `B2`, `W3`, `B3`.
2. The `ShaderProgram` for the decision field (`fieldShader`) is bound.
3. Uniform arrays are updated:
   - `fieldShader.setFloatArray(fieldW1Location, W1.data(), W1.size());`
   - and similarly for `B1`, `W2`, `B2`, `W3`, `B3`.
4. The `FieldVisualizer` draws a mesh covering the plot area.
5. For each fragment (pixel) of that mesh, the GPU runs the GLSL version of the network.

Dataset points are uploaded once per regeneration into a VBO in **`PointCloud`** and drawn each frame with `pointShader`.

### GPU side: field shader (`shaders/field.frag`)

The fragment shader mirrors the CPU NN math:

- Receives **uniform arrays**:
  - `u_W1[HIDDEN1 * INPUT_DIM]`, `u_b1[HIDDEN1]`.
  - `u_W2[HIDDEN2 * HIDDEN1]`, `u_b2[HIDDEN2]`.
  - `u_W3[OUTPUT_DIM * HIDDEN2]`, `u_b3[OUTPUT_DIM]`.
- Input `vPos` is the 2D position in the plot (clip-space / NDC), treated as `(x, y)`.
- Performs the same forward pass as `ToyNet`:
  - Dense layers with ReLU activations.
  - Softmax over 2 outputs, with a numerically stable logit shift.
- Computes `p1 = probs[1]` and uses it to mix between two base colors:
  - `c0` for class 0, `c1` for class 1.
- Writes a semi-transparent color to `FragColor`, overlaying the decision boundary under the points.

This makes the **decision field** update in real time as the CPU network trains, while evaluation is fully parallelized on the GPU fragment shader.

### GPU side: point shaders (`shaders/point.vert`, `shaders/point.frag`)

- **Vertex shader (`point.vert`)**

  - Inputs: `aPos` (2D position), `aLabel` (class label as float).
  - Outputs flat `vLabel` and `vIndex` to the fragment shader.
  - Sets `gl_Position` and `gl_PointSize` from `uPointSize`.

- **Fragment shader (`point.frag`)**
  - Uses `gl_PointCoord` to shape points as circles.
  - Colors point by label using `uColorClass0` / `uColorClass1`.
  - Compares `vIndex` to `uSelectedIndex` to draw an outline ring around the selected point.

Together, these shaders show **how vertex attributes, uniforms, and built-in variables** combine to render and highlight the dataset.

---

## Learning roadmap

If you want to use this repo to learn both NN math and the shader pipeline, a suggested path is:

1. **App and rendering setup**

   - Read `App.cpp` to see GLFW, GLAD, and ImGui initialization.
   - Inspect `ShaderProgram.cpp` to understand shader compilation, linking, and error logging.

2. **Neural network on CPU**

   - Study `ToyNet.h` / `ToyNet.cpp` for forward and backward passes.
   - Look at `Trainer.cpp` to see how batches are formed and how training progresses.

3. **Datasets and visualization**

   - Explore `DatasetGenerator.cpp` to see how each dataset is generated.
   - Read `FieldVisualizer` and `PointCloud` to see how geometry is uploaded and drawn.

4. **Neural network on GPU**

   - Carefully compare `ToyNet` with `shaders/field.frag`.
   - Note how shapes and indexing (`row * cols + col`) match between CPU and GPU.

5. **Interaction and UI**
   - Look at `ControlPanel.cpp` and `NetworkVisualizer.cpp` to understand the ImGui UI.
   - Read `Input.cpp` to see how mouse clicks are converted into NDC coordinates and used for sample selection.

As exercises, you can try:

- Changing hidden layer sizes (and matching GLSL constants) and watching the effect.
- Adding a new dataset to `DatasetGenerator` and exposing it in the UI.
- Modifying the color mapping in `field.frag` to visualize confidence or margins differently.

---

## License

MIT License

## Third-party Libraries

- [Dear ImGui](https://github.com/ocornut/imgui)
- [GLFW](https://www.glfw.org/)
- [GLAD](https://glad.dav1d.de/)
