using BandedMatrices
using Test, LinearAlgebra
using Random
using BandedPlusSemiseparableMatrices
import .BandedPlusSemiseparableMatrices: BandedPlusSemiseparableMatrix, BandedPlusSemiseparableQRPerturbedFactors, fast_qr!, onestep_qr!
#Random.seed!(1234)

n = 20
l = 4
m = 5
r = 2
p = 3
B = brandn(n,n,l,m)
U = randn(n,r)
V = randn(n,r)
W = randn(n,p)
S = randn(n,p)
A = BandedPlusSemiseparableQRPerturbedFactors(U,V,W,S,B)

F = qr(Matrix(A)).factors
Acopy = copy(Matrix(A))
mm, nn = size(Acopy)
τ_true = Vector{eltype(Acopy)}(undef, min(mm,nn))
LAPACK.geqrf!(Acopy, τ_true)

τ = fast_qr!(A)

@test Matrix(A) ≈ F
@test τ ≈ τ_true
