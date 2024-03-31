# DiffSL Language

The DSL is designed to be an easy and flexible language for specifying
DAE systems and is based on the idea that a DAE system can be specified by a set
of equations of the form:

$$
M(t) \frac{d\mathbf{u}}{dt} = F(\mathbf{u}, t)
$$

where $\mathbf{u}$ is the vector of state variables and $t$ is the time. The DSL
allows the user to specify the state vector $\mathbf{u}$ and the RHS function $F$.
Optionally, the user can also define the derivative of the state vector $d\mathbf{u}/dt$
and the mass matrix $M$ as a function of $d\mathbf{u}/dt$ (note that this function should be linear!).
The user is also free to define an an arbitrary number of intermediate
scalars and vectors of the users that are required to calculate $F$ and $M$.

## Defining variables

The DSL allows the user to define scalars, vectors, and dense/sparse/diagonal
n-dimensional tensors.  You can optionally label the elements of a vector or
tensor for later use.

For example, to define a scalar variable $k$ with value $1$, we write:

```
k { 1.0 }
```

To define a vector variable $\mathbf{v}$ with 3 elements that are labelled, we write:

```
v_i {
 x = 1.0,
 y = 2.0,
 z = 3.0,
}
```

The subscript `_i` indicates that this is a 1D vector, and $x$, $y$, and $z$ are
defined as labels to the 3 elements of the vector.  Later in the code, we could
refer to either the whole vector `v_i` for $\mathbf{v}$ or to the individual
elements $x$, $y$, and $z$, which are scalars.

To define dense 2x3 matrix $A$ with all elements set to `1.0`, we write:

```
A_ij {
 (0:2, 0:3) = 1.0,
}
```

Note the two subscript to indicate that this is a 2D tensor. The size of the
tensor is given in the brackets, and the elements are set to $1$.  If we have
additional rows, we can add them as follows:

```
A_ij {
 (0:2, 0:3) = 1.0,
 (3:4, 0:3) = 2.0,
}
```

We can define a sparse matrix $B$ by specifying the non-zero elements:

```
B_ij {
 (0, 0) = 1.0,
 (0, 1) = 2.0,
 (1, 1) = 3.0,
}
```

We can also define a diagonal identity matrix $I$ by specifying the diagonal
elements using a different range syntax:

```
I_ij {
 (0..2, 0..2) = 1.0,
}
```

## Operations

We can use standard algebraic operations on variables. To refer to previously
defined variables, we use the variable name, making sure to use the correct
subscript if it is a vector or tensor.

For example, to define a scalar variable $a$ as the sum of two other scalar
variables $b$ and $c$, we write:

```
a { b + c }
```

To define a vector variable $\mathbf{V}$ as the sum of two other vector
variables $\mathbf{u}$ and $\mathbf{w}$, we write:

```
v_i { u_i + w_i }
```

The indexing can be used to perform translations on tensors, for example the
following will define a new tensor $C$ that is the sum of $A$ and $B^T$:

```
C_ij { A_ij + B_ji }
```

Tensor indexing notation can also matrix-vector multiplications and any other
contraction operations. Any indices that do not appear in the output will be
summed over.  For example, the following will define a new vector $v$ that is
the result of a matrix-vector multiplication:

```
v_i { A_ij * u_j }
```

## Specifying inputs

We can override the values of any scalar variables by specifying them as input
variables.  To do this, we add a line at the top of the code to specify that
these are input variables:

```
in = [k]
k { 1.0 }
```

## Defining state variables

The primary goal of the DSL is to define a set of differential equations of a
system of state variables.  To define the state variables, we create a special
vector variable `u_i` which corresponds to the state variables $\mathbf{u}$.

The values that we use for `u_i` are the initial values of the state variables
at $t=0$.

```
u_i {
  x = 1,
  y = 0,
  z = 0,
}
```

We can optionally define the time derivatives of the state variables,
$\mathbf{\dot{u}}$ as well:

```
dudt_i {
  dxdt = 1,
  dydt = 0,
  dzdt = 0,
}
```

Here the initial values of the time derivatives are given, these are typically
used as a starting point to calculate a set of consistent initial values for the
state variables.

Note that there is no need to define `dudt` if you do not define a mass matrix $M$.

## Defining the ODE system equations

Recall that the DAE system is defined by the equations:

$$
M(t) \frac{d\mathbf{u}}{dt} = F(\mathbf{u}, t)
$$

We now define the equations $F$ and $M$ that we want to solve, using the
variables that we have defined earlier. We do this by defining a vector variable
`F_i` that corresponds to the RHS of the equations.

For example, to define a simple system of ODEs:

$$
\begin{align*}
 \frac{dx}{dt} &= y \\
 \frac{dy}{dt} &= -x \\
 x(0) &= 1 \\
 y(0) &= 0 \\
\end{align*}
$$

We write:

```
u_i {
 x = 1,
 y = 0,
}
F_i {
  y,
 -x,
}
```

We can also define a mass matrix $M$ by defining a vector variable `M_i`. This
is optional, and if not defined, the mass matrix is assumed to be the identity
matrix.

For example, lets define a simple DAE system using a singular mass matrix with a
zero on the diagonal:

$$
\begin{align*}
 \frac{dx}{dt} &= x \\
 0 &= y-x \\
 x(0) &= 1 \\
 y(0) &= 0 \\
\end{align*}
$$

We write:

```
u_i {
 x = 1,
 y = 0,
}
dudt_i {
 dxdt = 0,
 dydt = 1,
}
M_i {
 dxdt,
 0,
}
F_i {
 x,
 y-x,
}
```

## Specifying outputs

Finally, we specify the outputs of the system. These might be the state
variables themselves, or they might be other variables that are calculated from
the state variables. Here we specify that we want to output the state variables
$x$ and $y$:

```
out_i {
  x,
  y,
}
```

## Required variables

The DSL allows the user to specify an arbitrary number of intermediate
variables, but certain variables are required to be defined. These are:

* `u_i` - the state variables
* `F_i` - the vector $F(\mathbf{u}, t)$
* `out_i` - the output variables

## Predefined variables

The only predefined variable is the scalar $t$ which is the current time, this
allows the equations to be written as functions of time. For example

```
F_i {
  k1 * t + sin(t)
}
```

## Mathematical functions

The DSL supports the following mathematical functions:

* `pow(x, y)` - x raised to the power of y
* `sin(x)` - sine of x
* `cos(x)` - cosine of x
* `tan(x)` - tangent of x
* `exp(x)` - exponential of x
* `log(x)` - natural logarithm of x
* `sqrt(x)` - square root of x
* `abs(x)` - absolute value of x
* `sigmoid(x)` - sigmoid function of x