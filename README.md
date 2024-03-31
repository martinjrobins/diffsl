# DiffSL

A compiler for a domain-specific language for ordinary differential equations (ODEs) of the following form:

$$
M(t) \frac{d\mathbf{u}}{dt} = F(\mathbf{u}, t)
$$

As an example, the following code defines a classic DAE testcase, the Robertson
(1966) problem, which models the  kinetics of an autocatalytic reaction, given
by the following set of equations:

$$
\begin{align}
\frac{dx}{dt} &= -0.04x + 10^4 y z \\
\frac{dy}{dt} &= 0.04x - 10^4 y z - 3 \cdot 10^7 y^2 \\
0 &= x + y + z - 1
\end{align}
$$

The DiffSL code for this problem is as follows:


```
in = [k1, k2, k3]
k1 { 0.04 }
k2 { 10000 }
k3 { 30000000 }
u_i {
  x = 1,
  y = 0,
  z = 0,
}
dudt_i {
  dxdt = 1,
  dydt = 0,
  dzdt = 0,
}
M_i {
  dxdt,
  dydt,
  0,
}
F_i {
  -k1 * x + k2 * y * z,
  k1 * x - k2 * y * z - k3 * y * y,
  1 - x - y - z,
}
out_i {
  x,
  y,
  z,
}
```

## DiffSL Language Features

See the [DiffSL Language](#diffsl-language) section for a full description.

* Input parameters
* Define arbitrary intermediate variables:
  * Scalars (double precision floating point numbers)
  * Vectors (1D arrays of scalars)
  * N-dimensional tensor of scalars
  * Sparse tensors
* Tensor operations:
  * Elementwise operations
  * Tensor contractions/matmul/translation etc via index notation
* Stop the simulation when a condition is met

## Dependencies

You will need to install the [LLVM project](https://llvm.org/). The easiest way to
install this is to use the package manager for your operating system. For
example, on Ubuntu you can install these with the following command:

```bash
sudo apt-get install llvm
```

## Installing DiffSL

You can install DiffSL using cargo. You will need to indicate the llvm version you have installed using a feature flag. For example, for llvm 14:

```bash
cargo add diffsl --features llvm14-0
```

Other versions of llvm are also supported given by the features `llvm4-0`, `llvm5-0`, `llvm6-0`, `llvm7-0`, `llvm8-0`, `llvm9-0`, `llvm10-0`, `llvm11-0`, `llvm12-0`, `llvm13-0`, `llvm14-0`, `llvm15-0`, `llvm16-0`, `llvm17-0`.


