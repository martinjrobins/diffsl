# Defining tensor variables

The DiffSL language only has a single type of variable, which is a n-dimensional tensor filled with double precision floating point numbers.
These tensors can be dense, sparse or diagonal, and the compiler will try to choose the representation that is most efficient, preferring diagonal and then sparse matrices over dense.

The simplest tensor is a 0th dimensional scalar. For example, to define a scalar variable \\( k )\\$ with value 1.0, we write:

```
k { 1.0 }
```

Here `k` is the *label* for the tensor, which we can later use to refer to it in expressions. In the curly brackets we have one or more *elements* of the tensor,
and here `k` only has a single element, which is a constant value 1.

Lets now define a 1-dimensional vector variable $\mathbf{v}$ with 3 elements:

```
v_i {
 1.0,
 2.0,
 3.0,
}
```

The list of elements within a tensor are deliniated with a comma `,` and the trailing comma at the end of the list is optional.
Whitespace is ignored so you can also write this tensor all on the one line:

```
v_i { 1.0, 2.0, 3.0 }
```

## Subscripts

In the previous vector `v_i`, the subscript `_i` indicates that this is a 1D vector. Each subscript is a single character, 
and the number of subscripts indicates the number of dimensions of the tensor. You can use any character for the subscript,

```
v_x { 1.0, 2.0, 3.0 }
w_a { 2.0, 3.0 }
v_i { 1.0, 2.0, 3.0, 4.0 }
```

## Ranges

Each element of a tensor can optionally give a *range* of index numbers, which is used by the compiler to determine the extent of each element.
This is useful when defining higher dimensional tensors, such as matrices. For example, to define a 2x3 matrix $A$ with all elements set to `1.0`, we write:

```
A_ij {
 (0:2, 0:3) = 1.0,
}
```


Note the two subscript to indicate that this is a 2D tensor. The size of the
single element is given in the brackets, we have two ranges `0:2` and `0:3` that correspond to the two dimensions of the matrix.

Here is another example of a 4x2 matrix $B$ with rows 0 to 2 set to `1.0` and rows 3 to 4 set to `2.0`:

```
A_ij {
 (0:2, 0:3) = 1.0,
 (3:4, 0:3) = 2.0,
}
```

For specifying a single index, you can simply write the index number without the colon, for example to define a 3x3 identity matrix $I$:

```
I_ij {
 (0, 0) = 1.0,
 (1, 1) = 1.0,
 (2, 2) = 1.0,
}
```

Note that the compiler will automatically infer the size of the tensor from the ranges you provide, so you don't need to specify the size of the tensor explicitly.
Since the maximum index in the range is 2, the compiler will infer that the size of the tensor is 3x3.

Notice also that we have not defined all the elements of the matrix, only the non-zero elements. The compiler will assume that all other elements are zero.

Finally, you can also use the `..` operator to specify a *diagonal* range of indices. For example, to define a 3x3 matrix $D$ with the diagonal elements set to `1.0`:

```
D_ij {
 (0..2, 0..2) = 1.0,
}
```

## Sparse and diagonal matrices

We can automatically define a sparse matrix $B$ by simply specifying the non-zero elements:

```
B_ij {
 (0, 0) = 1.0,
 (0, 1) = 2.0,
 (1, 1) = 3.0,
}
```

The compiler will infer that this is a 2x2 matrix, and will automatically represent it as a sparse matrix.
We can force the compiler to use a dense representation by specifying the zeros explicitly:

```
B_ij {
 (0, 0) = 1.0,
 (0, 1) = 2.0,
 (1, 0) = 0.0,
 (1, 1) = 3.0,
}
```

As well as specifying a sparse matrix, we can also define a diagonal matrix by specifying the diagonal elements:

```
D_ij {
 (0, 0) = 1.0,
 (1, 1) = 2.0,
 (2, 2) = 3.0,
}
```

The compiler will infer that this is a 3x3 matrix, and will automatically represent it as a diagonal matrix.

## Labels

Each element of a tensor can optionally be given a name or *label* that can be used to refer to the element in expressions.

For example, to define a vector with two named elements:

```
v_i {
 x = 1.0,
 y = 2.0,
}
```

Here we have defined a single tensor `v_i` with two named elements `x` and `y`. We can then refer to the individual elements in expressions, where they will act as if they were separate variables:

```
v_i { x = 1.0, y = 2.0 }
w_i { 2 * y, 3 * x }
```


