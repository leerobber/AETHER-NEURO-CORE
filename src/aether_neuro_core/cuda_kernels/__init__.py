"""CUDA kernel sources for the HH engine.

`hh_solver.cu` is not compiled or invoked by any Python code in this package —
there is no CUDA toolchain or GPU available in the environment this repo was
authored in, so the kernel could not be built or run here. It is provided as
a from-scratch translation of `engines.hodgkin_huxley_vec`'s forward-Euler +
Euler-Maruyama math into CUDA C, intended as a starting point that still needs
real on-GPU verification (numerical parity against the NumPy reference, at
minimum) before being trusted.
"""
