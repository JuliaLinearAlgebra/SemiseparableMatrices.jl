using BandedMatrices
using Test, LinearAlgebra
using Random
import .BandedPlusSemiseparableMatrices: BandedPlusSemiseparableMatrix, BandedPlusSemiseparableQRPerturbedFactors

# test get index for BandedPlusSemiseparableQRPerturbedFactors

n = 10
l = 2
m = 3
r = 4
p = 5
B = brandn(n,n,l,m)
U = randn(n,r)
V = randn(n,r)
W = randn(n,p)
S = randn(n,p)
Q = randn(r,p)
K = randn(r,r)
Eₛ = randn(r,min(l+m,n))
Xₛ = randn(min(l,n),p)
Yₛ = randn(min(l,n),r)
Zₛ = randn(min(l,n),min(l+m,n))
A = BandedPlusSemiseparableQRPerturbedFactors(U,V,W,S,B)
A.Q[:,:] = Q
A.K[:,:] = K
A.Eₛ[:,:] = Eₛ
A.Xₛ[:,:] = Xₛ
A.Yₛ[:,:] = Yₛ
A.Zₛ[:,:] = Zₛ

E = [Eₛ zeros(r,max(n-(l+m),0))]
X = [Xₛ; zeros(max(n - l, 0),p)]
Y = [Yₛ; zeros(max(n - l, 0),r)]
Z = [Zₛ zeros(min(l,n),max(n-(l+m),0)); zeros(max(n-l,0),min(l+m,n)) zeros(max(n-l,0), max(n-(l+m),0))]

A₀ = tril(U*V',-1) + Matrix(B) + triu(W*S',1)
AA = A₀ + U*Q*S' + U*K*U'*A₀ + U*E + X*S' + Y*U'*A₀ + Z
@test Matrix(A) ≈ AA
