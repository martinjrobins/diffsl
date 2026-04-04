# Defining a system of ODEs

The primary goal of the DiffSL language is to define a system of ODEs in the following form:

$$
\begin{align*}
M(t) \frac{\mathrm{d}\mathbf{u}}{\mathrm{d}t} &= F_N(\mathbf{u}, \mathbf{p}, t) \\\\
\mathbf{u}(0) &= \mathbf{u}_0
\end{align*}
$$

where \\( \mathbf{u} \\) is the vector of state variables, \\( \mathbf{u}_0 \\) is the initial condition, \\( \mathbf{p} \\) is the parameter vector, \\( F_N \\) is the model-indexed RHS function, and \\( M \\) is the mass matrix.
The DSL allows the user to specify the state vector \\( \mathbf{u} \\), parameter vector \\( \mathbf{p} \\), and RHS function \\( F_N \\).

Optionally, the user can also define the derivative of the state vector \\( \mathrm{d}\mathbf{u}/\mathrm{d}t \\) and the mass matrix \\( M \\) as a function of \\( \mathrm{d}\mathbf{u}/\mathrm{d}t \\) 
(note that this function should be linear!). 
The user is also free to define an arbitrary number of intermediate tensors that are required to calculate \\( F \\) and \\( M \\). 


## Defining the state vector

To define the state vector \\(\mathbf{u} \\), we create a special
vector tensor `u_i`. Note the particular name `u` is used to indicate that this
is the state vector.

The values that we use for `u_i` are the initial values of the state variables
at \\( t=0 \\), so an initial condition \\( \mathbf{u}(t=0) = [x(0), y(0), z(0)] = [1, 0, 0] \\) is defined as:

```
u_i {
  x = 1,
  y = 0,
  z = 0,
}
```

Since we will often use the individual elements of the state vector in the RHS function, it is useful to define them as separate variables as well.

The state tensor `u` must be either be a scalar or a vector. So, if we only had a single state variable, we could define it as a scalar without the index `i` as follows:

```
u { x = 1 }
```

We can optionally define the time derivatives of the state variables,
\\( \mathbf{\dot{u}} \\) as well:

```
dudt_i {
  dxdt = 1,
  dydt = 0,
  dzdt = 0,
}
```

Here the initial values of the time derivatives are given.
In many cases any values can be given here as the time derivatives of the state variables are calculated from the RHS.
However, if there are any algebraic variables in the equations then these values can be used 
used as a starting point to calculate a set of consistent initial values for the
state variables.

Note that there is no need to define `dudt` if you do not define a mass matrix \\( M \\).

## Defining the ODE system equations

We now define the right-hand-side  function \\( F \\) that we want to solve, using the
variables that we have defined earlier. We do this by defining a vector variable
`F_i` that corresponds to the RHS of the equations.

For example, to define a simple system of ODEs:

$$
\begin{align*}
 \frac{dx}{dt} &= y \\\\
 \frac{dy}{dt} &= -x \\\\
 x(0) &= 1 \\\\
 y(0) &= 0 \\\\
\end{align*}
$$

We write:

```
u_i { x = 1, y = 0 }
F_i { y, -x }
```

## Using the Ode system index `N`

**Note: hybrid ODEs using `N` and `reset` are not yet supported in the current version of diffsol or pydiffsol, please check back for updates**

The index `N` is used to define multiple ODE systems in the same file, indexed by the non-negative integer `N`.
This allows us to define multiple ODE systems in the same file. Combined with the stop and reset functions (see below), 
this also allows us to define hybrid switching systems where we can switch between different ODE systems at different times during the simulation.

The model index `N` can be used in any of the equations, and it can be used to index into any of the tensors that we have defined.

For example, we can define two ODE systems, one with exponential growth and one with exponential decay, by
defining a `g_i` vector variable that contains the growth/decay rate for each system, and then using this variable in the definition of the RHS function `F_i`:

```
u_i { x = 1 }
g_i { 0.1, -0.1 }
F_i { g_i[N] * x }
```

## Stopping and resetting the ODE system

The `stop` and `reset` functions are standard tensors that can be used to stop and reset the ODE system at a given time, as long as the runtime supports this feature.

The `reset` tensor should be the same shape (i.e. a vector with the same number of elements) as the state vector `u_i`, whereas the `stop` tensor can be any length vector.
During the solve, whenever any element of the `stop` tensor is equal to zero, the ODE system will stop and, if the runtime supports hybrid models, the state of the 
ODE system will be reset to the values defined in the `reset` tensor, and the ODE system will continue to solve from there.

The classic example of this type of hybrid model is a bouncing ball, where the ODE system is stopped when the ball hits the ground (\(x \leq 0\)), and then the state of the system is reset to a new value that corresponds to the ball bouncing back up.

```
u_i {
 x = 0,
 v = 10,
}
F_i {
 v,
 -9.81,
}
stop {
 x,
}
reset {
 0,
 -0.8 * v,
}
```

Another example is a system that is periodically forced, for example dosing into a pharmacokinetic model, where the ODE system is stopped at regular intervals (e.g. every 24 hours), and the state of the system is reset to a new value that corresponds to the dose being administered.

```
u_i {
 x = 0,
}
F_i {
 -0.1 * x,
}
stop {
 t - 24,
}
reset {
 x + 10,
}
```

## Hybrid models with multiple ODE systems

The `stop` and `reset` functions can also be used to switch between different ODE systems defined in the same file using the model index `N`. The model index starts at `N = 0`, and every time a reset is triggered, the model index is set to the index of the `stop` that triggered the reset. This allows us to define hybrid models where we can switch between different ODE systems at different times during the simulation.

For example, we can define a system that starts with exponential growth, and then switches to exponential decay after 24 hours, and then switches back to exponential growth after another 24 hours, by defining two ODE systems as before, and then using the `stop` and `reset` functions to switch between them:


```
u_i { x = 1 }
g_i { 0.1, -0.1 }
F_i { g_i[N] * x }
stop { t - 48, t - 24 }
reset { 1 }
```

## Defining the mass matrix

We can also define a mass matrix \\( M \\) by defining a vector variable `M_i` which is the product of the mass matrix with the time derivative of the state vector \\( M \mathbf{\dot{u}} \\). 
This is optional, and if not defined, the mass matrix is assumed to be the identity
matrix.

Notice that we are defining a vector variable `M_i`, which is the LHS of the ODE equations \\( M \mathbf{\dot{u}} \\), and **not** the mass matrix itself.

For example, lets define a simple DAE system using a singular mass matrix with a
zero on the diagonal:

$$
\begin{align*}
 \frac{\mathrm{d}x}{\mathrm{d}t} &= x \\\\
 0 &= y-x \\\\
 x(0) &= 1 \\\\
 y(0) &= 0 \\\\
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