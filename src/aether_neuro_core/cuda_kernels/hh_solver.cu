/*
 * hh_solver.cu — vectorized Hodgkin-Huxley forward-Euler / Euler-Maruyama step.
 *
 * STATUS: UNVERIFIED. This file was written to mirror, term-for-term, the
 * Python reference implementation in
 * `aether_neuro_core/engines/hodgkin_huxley_vec.py` (classic squid-axon
 * kinetics, NEURON `hh.mod` parameterization). It has not been compiled or
 * run on a GPU — this environment has no CUDA toolchain or device available.
 * Before relying on it:
 *   1. Build it (nvcc) against a real toolchain.
 *   2. Run it against the same (V, m, h, n, i_ext) inputs as the NumPy engine
 *      and diff the resulting traces — do not trust it on inspection alone.
 *
 * Each CUDA thread owns exactly one neuron in the batch (one full (V, m, h, n)
 * state), matching the Python engine's per-neuron vectorization scheme. State
 * arrays are structure-of-arrays (SoA): threads access `V[i], m[i], h[i], n[i]`
 * where `i` is the global thread index, keeping memory access coalesced.
 *
 * Per-neuron parameters (g_na, g_k, g_leak, c_m, na_v_shift_mv, k_v_shift_mv)
 * are also SoA arrays, mirroring the Python engine's per-neuron parameter
 * broadcasting (a uniform sweep passes the same value in every slot).
 */

#include <math.h>

extern "C" {

/* vtrap(x, y) = x / (exp(x/y) - 1), with the x=0 singularity handled via its
 * analytic limit y*(1 - x/(2y)) — same trick as the Python `_vtrap` helper. */
__device__ __forceinline__ float vtrap(float x, float y) {
    float ratio = x / y;
    if (fabsf(ratio) < 1e-6f) {
        return y * (1.0f - ratio * 0.5f);
    }
    return x / (expf(ratio) - 1.0f);
}

__device__ __forceinline__ void hh_rates(
    float V,
    float na_shift,
    float k_shift,
    float* alpha_m, float* beta_m,
    float* alpha_h, float* beta_h,
    float* alpha_n, float* beta_n
) {
    float v_na = V - na_shift;
    float v_k = V - k_shift;

    *alpha_m = 0.1f * vtrap(-(v_na + 40.0f), 10.0f);
    *beta_m = 4.0f * expf(-(v_na + 65.0f) / 18.0f);
    *alpha_h = 0.07f * expf(-(v_na + 65.0f) / 20.0f);
    *beta_h = 1.0f / (expf(-(v_na + 35.0f) / 10.0f) + 1.0f);
    *alpha_n = 0.01f * vtrap(-(v_k + 55.0f), 10.0f);
    *beta_n = 0.125f * expf(-(v_k + 65.0f) / 80.0f);
}

__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

/*
 * hh_step_kernel: advance every neuron in the batch by one forward-Euler step,
 * with an additive Euler-Maruyama voltage-noise term (sigma * sqrt(dt) * noise[i]).
 *
 * `noise` must be pre-populated with iid standard-normal samples (e.g. via
 * cuRAND on the host/device before this kernel launches) — this kernel does
 * not itself generate random numbers, matching the Python engine's use of an
 * external `np.random.Generator`. Pass an all-zero `noise` buffer and
 * `sigma = 0` for the deterministic case.
 */
__global__ void hh_step_kernel(
    float* V, float* m, float* h, float* n,
    const float* i_ext,
    const float* g_na, const float* g_k, const float* g_leak,
    const float* c_m,
    const float* na_v_shift_mv, const float* k_v_shift_mv,
    const float* noise,
    float e_na, float e_k, float e_leak,
    float dt, float sigma,
    int batch_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    float v = V[i];
    float mi = m[i];
    float hi = h[i];
    float ni = n[i];

    float am, bm, ah, bh, an, bn;
    hh_rates(v, na_v_shift_mv[i], k_v_shift_mv[i], &am, &bm, &ah, &bh, &an, &bn);

    float i_na = g_na[i] * mi * mi * mi * hi * (v - e_na);
    float i_k = g_k[i] * ni * ni * ni * ni * (v - e_k);
    float i_leak = g_leak[i] * (v - e_leak);

    float dV = (i_ext[i] - i_na - i_k - i_leak) / c_m[i];
    float dm = am * (1.0f - mi) - bm * mi;
    float dh = ah * (1.0f - hi) - bh * hi;
    float dn = an * (1.0f - ni) - bn * ni;

    float voltage_noise = sigma * sqrtf(dt) * noise[i] / c_m[i];

    V[i] = v + dV * dt + voltage_noise;
    m[i] = clampf(mi + dm * dt, 0.0f, 1.0f);
    h[i] = clampf(hi + dh * dt, 0.0f, 1.0f);
    n[i] = clampf(ni + dn * dt, 0.0f, 1.0f);
}

} /* extern "C" */
