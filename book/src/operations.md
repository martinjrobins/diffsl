# Tensor Operations

We can use standard algebraic operations on variables. To refer to previously
defined variables, we use the variable name, making sure to use the correct
subscript if it is a vector or tensor.

For example, to define a scalar variable \\( a \\) as the sum of two other scalar
variables \\( b \\) and \\( c \\), we write:

```
b { 1.0 }
c { 2.0 }
a { b + c }
```

The scalar `a` will therefore be equal to 3.0.

To define a vector variable \\( \mathbf{v} \\) as the sum of two other vector
variables \\( \mathbf{u} \\) and \\( \mathbf{w} \\), we write:

```
u_i { 1.0, 2.0 }
w_i { 3.0, 4.0 }
v_i { u_i + w_i }
```

Notice that the index of the vectors within the expression must match the index of the output vector `v_i`.
So if we defined `v_a` instead of `v_i`, the expression would be:

```
v_a { u_a + w_a }
```

## Translations

For higher-dimensional tensors, the order of the indices in the expression relative to the output tensor is important.

For example, we can use the indices to define a translation of a matrix. Here we define a new matrix \\( C \\) that is the sum of \\( A \\) and \\( B^T \\)$,
where \\( B^T \\) is the transpose of \\( B \\)

```
C_ij { A_ij + B_ji }
```

Notice that the indices of \\( B^T \\) are reversed in the expression compared to the output tensor \\( C \\), indicating that we are indexing the rows of \\( B \\) with `j` and the columns with `i` when we calculate the `(i, j)` element of \\( C \\).

## Broadcasting

Broadcasting is supported in the language, so you can perform element-wise operations on tensors of different shapes. For example, the following will define a new vector \\( \mathbf{d} \\) that is the sum of \\( \mathbf{a} \\) and a scalar \\( k \\):

```
a_i { 1.0, 2.0 }
k { 3.0 }
d_i { a_i + k }
```

Here the scalar \\( k \\) is broadcast to the same shape as \\( \mathbf{a} \\) before the addition. The output vector \\( \mathbf{d} \\) will be \\( [4.0, 5.0] \\).

DiffSL uses the same broadcasting rules as NumPy, and you can read more about this in the [NumPy documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html).

## Contractions

The tensor indexing notation can also matrix-vector multiplications and any other
contraction operations. Any indices that do not appear in the output tensor will be
summed over.  For example, the following will define a new vector \\( \mathbf{v} \\) that is
the result of a matrix-vector multiplication:

```
v_i { A_ij * u_j }
```

Here the `j` index is summed over, so the `i` index of the output vector `v` is the sum of the element-wise product of the `i`th row of `A` and the vector `u`.

Another way to think about this matrix-vector multiplication is by considering it as a broadcasted element-wise multiplication followed by a sum or contraction over the `j` index. 
The `A_ij * u_j` expression broadcasts the vector `u` to the same shape as `A`, forming a new 2D tensor where each row is the element-wise product of the `i`th row of `A` and the vector `u`.
When we specify the output vector `v_i`, we implicitly then sum over the missing `j` index to form the final output vector.

For example, lets manually break this matrix-vector multiplication into two steps using an intermediary tensor `M_ij`:

```
M_ij { A_ij * u_j }
v_i { M_ij }
```

The first step calculates the element-wise product of `A` and `u` using broadcasting into the 2D tensor `M`, and the second step uses a contraction to sum over the `j` index to form the output vector `v`.


