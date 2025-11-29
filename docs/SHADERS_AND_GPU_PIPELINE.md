# Shaders and GPU Pipeline

This document focuses on how the project uses **OpenGL shaders** and the **GPU pipeline** to visualize the neural network.

The main ideas:

- The neural network is implemented on the **CPU** (for training) in `ToyNet`.
- The **same network** is mirrored on the **GPU** in the fragment shader `shaders/field.frag` to compute a decision field in real time.
- Point data is uploaded to vertex buffers and rendered with dedicated point shaders.

---

## 1. Quick OpenGL pipeline recap

At a high level, each frame the GPU does:

1. **Vertex processing** via a vertex shader.
   - Input: vertex attributes (positions, labels, etc.).
   - Output: clip-space positions and per-vertex varyings.
2. **Rasterization** produces fragments (potential pixels) from primitives (triangles, lines, points).
3. **Fragment processing** via a fragment shader.
   - Input: interpolated varyings and uniforms.
   - Output: final color for each fragment.

In this project, the NN math lives in the **fragment shader** for the decision field.

---

## 2. Data flow for the decision field

The decision field (colored background) is drawn by:

- CPU: `App::renderLoop` and `FieldVisualizer`.
- GPU: `field.vert` (simple geometry) + `field.frag` (neural network).

### 2.1 CPU side: uploading network parameters

In `App::renderLoop` (see `src/core/App.cpp`):

1. The current neural network parameters are taken from `Trainer::net`:

   ```cpp
   const auto& W1 = trainer.net.getW1();
   const auto& B1 = trainer.net.getB1();
   const auto& W2 = trainer.net.getW2();
   const auto& B2 = trainer.net.getB2();
   const auto& W3 = trainer.net.getW3();
   const auto& B3 = trainer.net.getB3();
   ```

2. The field shader is bound and uniform arrays are set:

   ```cpp
   fieldShader.use();
   fieldShader.setFloatArray(fieldW1Location, W1.data(), (int)W1.size());
   fieldShader.setFloatArray(fieldB1Location, B1.data(), (int)B1.size());
   fieldShader.setFloatArray(fieldW2Location, W2.data(), (int)W2.size());
   fieldShader.setFloatArray(fieldB2Location, B2.data(), (int)B2.size());
   fieldShader.setFloatArray(fieldW3Location, W3.data(), (int)W3.size());
   fieldShader.setFloatArray(fieldB3Location, B3.data(), (int)B3.size());
   ```

3. `FieldVisualizer::draw()` renders a mesh (typically a grid of quads) covering the plot region. For each fragment of this mesh, the GPU runs `field.frag`.

The key point: **all weights and biases are packed into uniform arrays** so that the fragment shader can run the same forward pass as `ToyNet`.

### 2.2 GPU side: `field.frag`

The shader `shaders/field.frag` declares:

```glsl
const int INPUT_DIM  = 2;
const int HIDDEN1    = 4;
const int HIDDEN2    = 8;
const int OUTPUT_DIM = 2;

uniform float u_W1[HIDDEN1 * INPUT_DIM];
uniform float u_b1[HIDDEN1];
uniform float u_W2[HIDDEN2 * HIDDEN1];
uniform float u_b2[HIDDEN2];
uniform float u_W3[OUTPUT_DIM * HIDDEN2];
uniform float u_b3[OUTPUT_DIM];
```

It receives `vPos`, the 2D position of the fragment in clip/NDC space, and then performs the **same math** as `ToyNet`:

1. Treat `vPos` as the input `a0`.
2. Compute layer 1 pre-activations and ReLU activations using `u_W1` and `u_b1`.
3. Compute layer 2 with `u_W2` and `u_b2`.
4. Compute output logits and softmax using `u_W3` and `u_b3`.
5. Use `p1 = probs[1]` to mix between two base colors `c0` and `c1`.

Because this runs for **every fragment in parallel**, you get a dense, smooth decision field that updates in real time as training progresses.

---

## 3. Data flow for the point cloud

Dataset points themselves are rendered using a different shader pair: `point.vert` and `point.frag`.

### 3.1 CPU side: `PointCloud`

The `PointCloud` class (see `FieldVisualizer` / `PlotGeometry` usage) manages:

- A vertex buffer object (VBO) storing:
  - 2D position `aPos` for each data point.
  - Class label as a float `aLabel`.
- A vertex array object (VAO) describing how to interpret the buffer:
  - `layout(location = 0)` → `aPos`.
  - `layout(location = 1)` → `aLabel`.

When the dataset is regenerated, the CPU side uploads all points into this VBO once via `glBufferData` or `glBufferSubData`. After that, each frame simply re-binds the VAO and issues a draw call.

### 3.2 GPU side: `point.vert`

`shaders/point.vert`:

```glsl
layout (location = 0) in vec2 aPos;
layout (location = 1) in float aLabel;

flat out int vLabel;
flat out int vIndex;

uniform float uPointSize;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vLabel = int(aLabel + 0.5);
    vIndex = gl_VertexID;
    gl_PointSize = uPointSize;
}
```

- `aPos` is passed directly to `gl_Position` (already in clip/NDC space).
- `aLabel` is converted to an integer and sent to the fragment shader.
- `gl_VertexID` is used as a unique point index (`vIndex`), which allows the fragment shader to know which point is being drawn.

### 3.3 GPU side: `point.frag`

`shaders/point.frag`:

```glsl
flat in int vLabel;
flat in int vIndex;

uniform vec3 uColorClass0;
uniform vec3 uColorClass1;
uniform int  uSelectedIndex;

void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    float r2 = dot(d, d);
    if (r2 > 0.25) discard;  // circle mask

    bool isSelected = (uSelectedIndex >= 0 && vIndex == uSelectedIndex);
    if (isSelected && r2 > 0.16 && r2 < 0.25) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0); // outline ring
        return;
    }

    vec3 color = (vLabel == 0) ? uColorClass0 : uColorClass1;
    FragColor = vec4(color, 1.0);
}
```

This shader:

- Shapes the point into a disc using `gl_PointCoord`.
- Colors it based on the class label.
- Draws an outline ring around the selected point by comparing `vIndex` with `uSelectedIndex`.

Selection and probe logic are handled on the CPU side in `Input.cpp` and `ControlPanel.cpp`. When you click near a point, the CPU:

- Finds the closest point in NDC space.
- Updates `UiState` (probe position, selected index, label).
- Uploads `uSelectedIndex` as a uniform so this shader can highlight it.

---

## 4. ImGui overlays vs GPU drawing

Dear ImGui is used for:

- The "Neural Net Controls" window.
- The network diagram.
- Loss/accuracy plots.

ImGui draws **after** the main OpenGL scene:

1. The field and points are drawn using the custom shaders above.
2. ImGui builds its own vertex buffers and issues its own draw calls.

Conceptually, ImGui is just another client of the same GPU pipeline, but you usually don’t write its shaders yourself; they are provided by the ImGui backends (`imgui_impl_glfw`, `imgui_impl_opengl3`).

---

## 5. Relation to "real" deep learning GPU usage

This project uses the GPU in a **visualization-oriented** way rather than a high-throughput training engine. Still, many conceptual links to deep learning GPU usage are visible:

- **Parallelism**

  - In typical DL training, a big matrix multiplication (like `W · A`) is executed in parallel over many elements using cuBLAS or similar libraries.
  - Here, many **fragments** are processed in parallel, each running a small copy of the network for its `(x, y)` position.

- **Data layout**

  - Both CPU (`ToyNet`) and GPU (`field.frag`) store weights in flat arrays using the same index formula (`row * cols + col`).
  - This makes it straightforward to send parameters from CPU to GPU without reshaping.

- **Inference vs training**

  - On the CPU, `ToyNet::trainBatch` does both forward and backward passes and updates parameters.
  - On the GPU, `field.frag` performs **inference only** (no gradients, no updates).

- **Batching**
  - Real DL code typically evaluates many samples at once on the GPU.
  - In this project, each fragment is like a tiny independent sample; the “batch” is the set of pixels covering the field. This is great for visualization but would be inefficient for large-scale training.

This design is intentionally simple so you can match each term in the CPU math to a corresponding term in GLSL.

---

## 6. Performance notes and limitations

- The network size is small (4 and 8 hidden units), which keeps uniform arrays small and shader code simple.
- All weights are uploaded each frame, which is fine here but would be expensive for very large networks.
- Uniform arrays have size limits depending on the GPU; very large networks would need textures or SSBOs instead.
- Evaluating a network in the fragment shader is perfect for visualizations like decision fields, but for real training/inference engines you would typically:
  - Use compute shaders, CUDA, or another GPGPU API.
  - Use highly optimized libraries for matrix multiplications.

---

## 7. How to experiment

If you want to explore the shader/GPU side further:

1. **Change network width**

   - Increase `Hidden1` and/or `Hidden2`.
   - Update the constants in `ToyNet` and in `field.frag`.
   - Make sure uniform array sizes and loops match.

2. **Visualize different quantities**

   - Instead of just `p1`, visualize:
     - `p0 − p1` (margin), mapped to a diverging colormap.
     - The pre-activation `z2` of a specific hidden unit.

3. **Animate parameters**

   - Temporarily freeze training and interpolate between two sets of weights on the CPU, uploading the interpolated weights each frame.

4. **Use different color maps**
   - Modify the color mix in `field.frag` to use non-linear ramps or discrete contour bands.

By tinkering with shaders while keeping the core math the same, you can build intuition for how GPUs execute neural‑network-style computations in a massively parallel graphics context.
