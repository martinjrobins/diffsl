# Pre-defined functions & variables

## Pre-defined functions

The DiffSL supports the following mathematical functions that can be used in an expression:

* `pow(x, y)` - x raised to the power of y
* `sin(x)` - sine of x
* `cos(x)` - cosine of x
* `tan(x)` - tangent of x
* `exp(x)` - exponential of x
* `tanh(x)` - hyperbolic tangent of x
* `sinh(x)` - hyperbolic sine of x
* `cosh(x)` - hyperbolic cosine of x
* `arcsinh(x)` - inverse hyperbolic sine of x
* `arccosh(x)` - inverse hyperbolic cosine of x
* `log(x)` - natural logarithm of x
* `log10(x)` - base-10 logarithm of x
* `sqrt(x)` - square root of x
* `abs(x)` - absolute value of x
* `sigmoid(x)` - sigmoid function of x
* `heaviside(x)` - Heaviside step function of x
* `interp1d(x_i, y_i, q)` - 1D piecewise linear interpolation

You can use these functions as part of an expression in the DSL. For example, to define a variable `a` that is the sine of another variable `b`, you can write:

```diffsl
b { 1.0 }
a { sin(b) }
```

### `interp1d(x_i, y_i, q)`

The `interp1d` function performs 1D piecewise linear interpolation. `x_i` and `y_i`
must be compile-time constant 1D dense tensors of equal length. The query point
`q` can be a scalar or a tensor. Out-of-range queries are clamped to the
endpoints.

`x_i` must be **strictly increasing**; an error is reported at compile time if
it is not.

If `x_i` is uniformly spaced (constant increment `dx` between consecutive values),
the interpolation runs in **O(1)** per query point.  For non-uniform `x_i`, a
binary search is used, giving **O(log n)** cost per query point.

```diffsl
xs_i { 0.0, 10.0, 20.0 }
ys_i { 1.0,  5.0, 15.0 }
r { interp1d(xs_i, ys_i, 5.0) }     # => 3.0

# batched queries
q_j { 0.0, 5.0, 10.0, 15.0, 20.0 }
r_j { interp1d(xs_i, ys_i, q_j) }   # => [1.0, 3.0, 5.0, 10.0, 15.0]

# the first two arguments may be any constant 1D dense expression
r { interp1d(xs_i * 2.0, ys_i + 1.0, 5.0) }
```

## Pre-defined variables

There are two predefined variables in DiffSL. The first is the scalar `t` which is the current time, this allows the equations to be written as functions of time. For example

```diffsl
F_i { t + sin(t) }
```

The second is the scalar `N` which is the current model index, this allows us to write hybrid models with multiple ODE systems in the same file. For example

```diffsl
F_i { g_i[N] * x }
```
