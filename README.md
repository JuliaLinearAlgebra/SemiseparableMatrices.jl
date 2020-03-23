# SemiseparableMatrices.jl
A Julia package to represent semiseparable and almost banded matrices


## `SemiseparableMatrix`

A semiseparable matrix of semiseparability rank `r` has the form
```julia
tril(L) + triu(U)
``` 
where `L` and `U` are rank-`r` matrices. We can construct a semiseparable matrix as follows:
```julia

```


Remark: There is a more general definition of semiseparability sometimes used based on rank of off-diagonal blocks. Such matrices would be better suited represented as hierarchical matrices as in [HierarchicalMatrices.jl](https://github.com/JuliaMatrices/HierarchicalMatrices.jl).

## `AlmostBandedMatrix`

An almost-banded matrix has the form
```julia
B + triu(U,u+1)
```
where `B` is a banded matrix with bandwidths `(l,u)` and `U` is a rank-`r` matrix.
