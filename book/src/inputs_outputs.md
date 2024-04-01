# Inputs & Outputs

Often it is useful to parameterize the system of equations using a set of input parameters. It is also useful to be able to extract certain variables from the system for further analysis. 
In this section we will show how to specify inputs and outputs in the DiffSL language.

## Specifying inputs

We can override the values of any scalar variables by specifying them as input
variables.  To do this, we add a line at the top of the code to specify that
these are input variables:

```
in = [k]
k { 1.0 }
u { 0.1 }
F { k * u }
```

Here we have specified a single input parameter `k` that is used in the RHS function `F`. 
The value of `k` is set to `1.0` in the code, but this value is only a default, and can be overridden by passing in a value at solve time.

We can use input parameters anywhere in the code, including in the definition of other input parameters.

```
in = [k]
k { 1.0 }
g { 2 * k }
F { g * u }
```

or in the intial conditions of the state variables:

```
in = [k]
k { 1.0 }
u_i {
  x = k,
}
F { u }
```


## Specifying outputs

We can also specify the outputs of the system. These might be the state
variables themselves, or they might be other variables that are calculated from
the state variables. 

Here is an example where we simply output the elements of the state vector:

```
u_i {
  x = 1.0,
  y = 2.0,
  z = 3.0,
}
out_i { x, y, z }
```

or we can derive additional outputs from the state variables:

```
u_i {
  x = 1.0,
  y = 2.0,
  z = 3.0,
}
out { x + y + z }
```