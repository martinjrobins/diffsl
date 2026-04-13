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

You can use these functions as part of an expression in the DSL. For example, to define a variable `a` that is the sine of another variable `b`, you can write:

```diffsl
b { 1.0 }
a { sin(b) }
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
