DiffSL Language

The DSL is designed to be an easy and flexible language for specifying
DAE systems and is based on the idea that a DAE system can be specified by a set
of equations of the form:

$$
M(t) \frac{d\mathbf{u}}{dt} = F(\mathbf{u}, t)
$$

where \\( \mathbf{u}$ \\) is the vector of state variables and \\( t \\) is the time. The DSL
allows the user to specify the state vector \\( \mathbf{u} \\) and the RHS function \\( F \\).
Optionally, the user can also define the derivative of the state vector \\( d\mathbf{u}/dt \\)
and the mass matrix \\( M \\) as a function of \\( d\mathbf{u}/dt \\) (note that this function should be linear!).

The user is also free to define an an arbitrary number of intermediate
scalars and vectors of the users that are required to calculate \\( F \\) and \\( M \\).

## A Simple Example

To illustrate the language, consider the following simple example of a logistic growth model:

$$
\frac{dN}{dt} = r N (1 - N/K)
$$

where \\( N \\) is the population, \\( r \\) is the growth rate, and \\( K \\) is the carrying capacity.

To specify this model in DiffSL, we can write:

``
in = [r, k]
u_i {
  N = 0.0
}
F_i {
  r * N * (1 - N/k)
}
out_i {
  N
}
``

Here, we define the input parameters for our model as a vector `in` with the growth rate `r` and the carrying capacity `k`. We then define the state vector `u_i` with the population `N` initialized to `0.0`. Next, we define the RHS function `F_i` as the logistic growth equation. Finally, we define the output vector `out_i` with the population `N`.



