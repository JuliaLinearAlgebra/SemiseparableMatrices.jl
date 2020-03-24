# SemiseparableMatrices.jl
A Julia package to represent semiseparable and almost banded matrices

[![Build Status](https://travis-ci.org/JuliaMatrices/SemiseparableMatrices.jl.svg?branch=master)](https://travis-ci.org/JuliaMatrices/SemiseparableMatrices.jl) 

[![codecov](https://codecov.io/gh/JuliaMatrices/SemiseparableMatrices.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaMatrices/SemiseparableMatrices.jl)



## `SemiseparableMatrix`

A semiseparable matrix of semiseparability rank `r` has the form
```julia
tril(A,-l-1) + triu(B,u+1)
``` 
where `A` and `B` are rank-`r` matrices. We can construct a semiseparable matrix as follows:
```julia
julia> using SemiseparableMatrices, FillArrays

julia> n = 5;

julia> A = Fill(1.0,n,n);

julia> B = Fill(2.0,n,n);

julia> SemiseparableMatrix(A, B, 2, 0)
5×5 SemiseparableMatrix{Float64,Fill{Float64,2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}},Fill{Float64,2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}}}:
 2.0  2.0  2.0  2.0  2.0
 0.0  2.0  2.0  2.0  2.0
 1.0  0.0  2.0  2.0  2.0
 1.0  1.0  0.0  2.0  2.0
 1.0  1.0  1.0  0.0  2.0
```
It is advised to use a low rank type for `A` and `B`: here we use `Fill` from FillArrays.jl, though another good option is `ApplyArray(*, U, V)` from
LazyArrays.jl.

Remark: There is a more general definition of semiseparability sometimes used based on rank of off-diagonal blocks. Such matrices would be better suited represented as hierarchical matrices as in [HierarchicalMatrices.jl](https://github.com/JuliaMatrices/HierarchicalMatrices.jl).

## `AlmostBandedMatrix`

An almost-banded matrix has the form
```julia
A + triu(B,u+1)
```
where `A` is a banded matrix with bandwidths `(l,u)` and `B` is a rank-`r` matrix.
These arise in discretizations of spectral methods. For example, if we wish to
solve the ODE for computing `exp(z)` with Taylor series, applying the boundary condition
at `z = 1`, we arrive at an almost-banded system:
```julia
julia> n = 20; A = AlmostBandedMatrix(BandedMatrix(0 => [1; 1:n-1], -1 => Fill(-1.0,n-1)), ApplyMatrix(*, [1; zeros(n-1)], Fill(1.0,1,n)))
20×20 AlmostBandedMatrix{Float64,Array{Float64,2},Array{Float64,1},Fill{Float64,2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}},Base.OneTo{Int64}}:
  1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0
 -1.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  0.0  -1.0   2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  0.0   0.0  -1.0   3.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  0.0   0.0   0.0  -1.0   4.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  0.0   0.0   0.0   0.0  -1.0   5.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  0.0   0.0   0.0   0.0   0.0  -1.0   6.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  0.0   0.0   0.0   0.0   0.0   0.0  -1.0   7.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  0.0   0.0   0.0   0.0   0.0   0.0   0.0  -1.0   8.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -1.0   9.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -1.0  10.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -1.0  11.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -1.0  12.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -1.0  13.0   0.0   0.0   0.0   0.0   0.0   0.0
  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -1.0  14.0   0.0   0.0   0.0   0.0   0.0
  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -1.0  15.0   0.0   0.0   0.0   0.0
  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -1.0  16.0   0.0   0.0   0.0
  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -1.0  17.0   0.0   0.0
  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -1.0  18.0   0.0
  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -1.0  19.0

julia> A \ [exp(1); zeros(n-1)] # Taylor coefficients of exp(z)
20-element Array{Float64,1}:
 0.9999999999999999    
 1.0000000000000002    
 0.5000000000000002    
 0.16666666666666674   
 0.04166666666666668   
 0.008333333333333337  
 0.0013888888888888898 
 0.00019841269841269855
 2.4801587301587315e-5 
 2.755731922398591e-6  
 2.755731922398591e-7  
 2.5052108385441733e-8 
 2.0876756987868117e-9 
 1.6059043836821624e-10
 1.1470745597729734e-11
 7.647163731819823e-13 
 4.7794773323873897e-14
 2.811457254345522e-15 
 1.5619206968586235e-16
 8.220635246624334e-18 
```
Note that solving linear systems is O(n) complexity here, so the above works even with `n = 1_000_000`, solving
in just over a second.


# References

[1] Chandrasekaran and Gu, Fast and stable algorithms for banded plus
Semiseparable systems of linear equations, SIMAX, 25 (2003), pp. 373-384.