# Pre-defined functions

The DiffSL supports the following mathematical functions that can be used in an expression:

* `pow(x, y)` - x raised to the power of y
* `sin(x)` - sine of x
* `cos(x)` - cosine of x
* `tan(x)` - tangent of x
* `exp(x)` - exponential of x
* `log(x)` - natural logarithm of x
* `sqrt(x)` - square root of x
* `abs(x)` - absolute value of x
* `sigmoid(x)` - sigmoid function of x
* `heaviside(x)` - Heaviside step function of x

You can use these functions as part of an expression in the DSL. For example, to define a variable `a` that is the sine of another variable `b`, you can write:

```
b { 1.0 }
a { sin(b) }
```

# Pre-defined variables

The only predefined variable is the scalar `t` which is the current time, this allows the equations to be written as functions of time. For example

```
F_i {
  t + sin(t)
}
```