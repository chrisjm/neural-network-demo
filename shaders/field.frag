#version 330 core
in vec2 vPos;
out vec4 FragColor;
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
void main()
{
    float a0[INPUT_DIM];
    a0[0] = vPos.x;
    a0[1] = vPos.y;
    float a1[HIDDEN1];
    for (int j = 0; j < HIDDEN1; ++j) {
        float sum = u_b1[j];
        for (int i = 0; i < INPUT_DIM; ++i) {
            int idx = j * INPUT_DIM + i;
            sum += u_W1[idx] * a0[i];
        }
        a1[j] = max(sum, 0.0);
    }
    float a2[HIDDEN2];
    for (int j = 0; j < HIDDEN2; ++j) {
        float sum = u_b2[j];
        for (int i = 0; i < HIDDEN1; ++i) {
            int idx = j * HIDDEN1 + i;
            sum += u_W2[idx] * a1[i];
        }
        a2[j] = max(sum, 0.0);
    }
    float logits[OUTPUT_DIM];
    float maxLogit = -1e30;
    for (int k = 0; k < OUTPUT_DIM; ++k) {
        float sum = u_b3[k];
        for (int j = 0; j < HIDDEN2; ++j) {
            int idx = k * HIDDEN2 + j;
            sum += u_W3[idx] * a2[j];
        }
        logits[k] = sum;
        if (sum > maxLogit) maxLogit = sum;
    }
    float expSum = 0.0;
    float probs[OUTPUT_DIM];
    for (int k = 0; k < OUTPUT_DIM; ++k) {
        float e = exp(logits[k] - maxLogit);
        probs[k] = e;
        expSum += e;
    }
    if (expSum <= 0.0) {
        FragColor = vec4(0.5, 0.5, 0.5, 0.4);
        return;
    }
    for (int k = 0; k < OUTPUT_DIM; ++k) {
        probs[k] /= expSum;
    }
    float p1 = probs[1];
    vec3 c0 = vec3(0.2, 0.6, 1.0);
    vec3 c1 = vec3(1.0, 0.5, 0.2);
    vec3 color = mix(c0, c1, p1);
    FragColor = vec4(color, 0.4);
}
