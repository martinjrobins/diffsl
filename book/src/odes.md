# Defining a system of ODEs

The primary goal of the DiffSL language is to define a system of ODEs in the following form:

$$
\begin{align*}
M(t) \frac{\mathrm{d}\mathbf{u}}{\mathrm{d}t} &= F(\mathbf{u}, t) \\\\
\mathbf{u}(0) &= \mathbf{u}_0
\end{align*}
$$

where \\( \mathbf{u} \\) is the vector of state variables, \\( \mathbf{u}_0 \\) is the initial condition, \\( F \\) is the RHS function, and \\( M \\) is the mass matrix. 
The DSL allows the user to specify the state vector \\( \mathbf{u} \\) and the RHS function \\( F \\). 

Optionally, the user can also define the derivative of the state vector \\( \mathrm{d}\mathbf{u}/\mathrm{d}t \\) and the mass matrix \\( M \\) as a function of \\( \mathrm{d}\mathbf{u}/\mathrm{d}t \\) 
(note that this function should be linear!). 
The user is also free to define an arbitrary number of intermediate tensors that are required to calculate \\( F \\) and \\( M \\). 


## Defining state variables

To define the state variables \\(\mathbf{u} \\), we create a special
vector variable `u_i`. Note the particular name `u` is used to indicate that this
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