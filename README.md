# DiffSL

A compiler for a domain-specific language for ordinary differential equations (ODE) and differential algebraic equations (DAE).

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
F_i {
  dxdt,
  dydt,
  0,
}
G_i {
  -k1 * x + k2 * y * z,
  k1 * x - k2 * y * z - k3 * y * y,
  1 - x - y - z,
}
out_i {
  x,
  y,
  z,
```

## Installation

This package relies on the `clang` (https://clang.llvm.org/) and `opt`
executables from the [LLVM project](https://llvm.org/). The easiest way to
install these is to use the package manager for your operating system. For
example, on Ubuntu you can install these with the following command:

```bash
sudo apt-get install clang
```

In addition, DiffSL uses the [Enzyme AD](https://enzyme.mit.edu/) package for automatic differentiation. This can be installed by following the instructions on the Enzyme AD website. You will need set the `LIBRARY_PATH` environment variable to the location of the Enzyme AD library. For example, if you have installed Enzyme AD in the directory `/usr/local`, you can set the `LIBRARY_PATH` environment variable with the following command:

```bash
export LIBRARY_PATH=/usr/local/lib
```

## Building DiffSL






### Installing Enzyme AD

```bash
cmake -DCMAKE_INSTALL_PREFIX=<install> -DCMAKE_BUILD_TYPE=Release  ..
```

## DiffSL Language

The DSL is designed to be an easy and flexible language for specifying
DAE systems and is based on the idea that a DAE system can be specified by a set
of equations of the form:

$$F(\mathbf{u}, \mathbf{\dot{u}}, t) = G(\mathbf{u}, t)$$

where $\mathbf{u}$ is the vector of state variables, $\mathbf{\dot{u}}$ is the
vector of time derivatives of the state variables, and $t$ is the time. The DSL
allows the user to specify the state vector $\mathbf{u}$ and the vector of time
derivatives of the state vector $\mathbf{\dot{u}}$, vectors $F$ and $G$
calculated at each timestep, along with an arbitrary number of intermediate
scalars and vectors of the users that are required to calculate $F$ and $G$.

### Defining variables

To write down the robertson problem given above, we first define some scalar
variables that we will use in the equations:

```
k1 { 0.04 }
k2 { 10000 }
k3 { 30000000 }
```

The names `k1`, `k2`, and `k3` are arbitrary names and can be used to refer to
the values of these scalars. The values themselves are given within the curly
braces `{}`. Here they are given as constant values, but they could also be
given as functions of time (e.g. `k1 { 0.04 * sin(t) }`), or as functions of the
other variables in the system (e.g. `k2 { 10 * k1 }`).

### Specifying inputs

We also want to potentially vary these values and resolve the system for
different values of `k1`, `k2`, and `k3`. To do this, we add a line at the top
of the code to specify that these are input variables:

```
in = [k1, k2, k3]
```

### Defining state variables

Next we define the state variables of the system, $\mathbf{u}$, and their initial values

```
u_i {
  x = 1,
  y = 0,
  z = 0,
}
```

Here `u` is the name of the vector of state variables, and the subscript `_i`
indicates that this is a 1D vector (notice how `k1` etc. do not have a subscript
as they are defined as scalars) , and `x`, `y`, and `z` are defined as labels to
the 3 elements of the vector. The values of the state variables at the initial
time are given after the `=` sign.

We next define the time derivatives of the state variables, $\mathbf{\dot{u}}$:

```
dudt_i {
  dxdt = 1,
  dydt = 0,
  dzdt = 0,
}
```

Here the initial values of the time derivatives are given, but for dxdt and dydt
this initial value is not used as we give explicit equations for these below in
the `F_i` and `G_i` sections. For the third element of the vector, `dzdt`, the
initial value is used as a starting point to calculate a set of consistent
initial values for the state variables.

### Defining the DAE system equations

We now define the equations $F$ and $G$ that we want to solve, using the
variables that we have defined above, both the input parameters and the state
variables. 


```
F_i {
  dxdt,
  dydt,
  0,
}
G_i {
  -k1 * x + k2 * y * z,
  k1 * x - k2 * y * z - k3 * y * y,
  1 - x - y - z,
}
```

### Specifying outputs

Finally, we specify the outputs of the system. These might be the state
variables themselves, or they might be other variables that are calculated from
the state variables. Here we specify that we want to output the state variables
`x`, `y`, and `z`:

```
out_i {
  x,
  y,
  z,
}
```


### Required variables

The DSL allows the user to specify an arbitrary number of intermediate variables, but certain variables are required to be defined. These are:

* `u_i` - the state variables
* `dudt_i` - the time derivatives of the state variables
* `F_i` - the vector $F(\mathbf{u}, \mathbf{\dot{u}}, t)$
* `G_i` - the vector $G(\mathbf{u}, t)$
* `out_i` - the output variables

### Predefined variables

The only predefined variable is the scalar `t` which is the current time, this allows the equations to be written as functions of time. For example

```
F_i {
  dydt,
}
G_i {
  k1 * t + sin(t)
}
```

### Mathematical functions

The DSL supports the following mathematical functions:

* `sin(x)` - sine of x
* `cos(x)` - cosine of x
* `tan(x)` - tangent of x
* `exp(x)` - exponential of x
* `log(x)` - natural logarithm of x
* `sqrt(x)` - square root of x
* `abs(x)` - absolute value of x
* `sigmoid(x)` - sigmoid function of x