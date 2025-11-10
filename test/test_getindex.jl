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
jj = 3
E_j = [Eₛ zeros(r,max(n-jj+1-(l+m),0))]
X_j = [Xₛ; zeros(max(n -jj+1- l, 0),p)]
Y_j = [Yₛ; zeros(max(n -jj+1- l, 0),r)]
Z_j = [Zₛ zeros(min(l,n),max(n-jj+1-(l+m),0)); zeros(max(n-jj+1-l,0),min(l+m,n)) zeros(max(n-jj+1-l,0), max(n-jj+1-(l+m),0))]
AA2 = A₀[jj:end,jj:end] + U[jj:end,:]*Q*(S[jj:end,:])' + U[jj:end,:]*K*(U[jj:end,:])'*A₀[jj:end,jj:end]+
      U[jj:end,:]*E_j + X_j*(S[jj:end,:])' + Y_j*(U[jj:end,:])'*A₀[jj:end,jj:end] + Z_j

@test Matrix(A) ≈ AA

j = 2
UᵀA = zeros(r, n+1-j)
UᵀU = (U[j:end,:])'*U[j:end,:]
UᵀW = zeros(r,p)
for i in j:n
    UᵀU[:,:] -= U[i,:] * (U[i,:])'
    UᵀA[:,i+1-j] = UᵀU*V[i,:] + (U[max(j,i-m) : min(i+l,n),:])'*B[max(j,i-m) : min(i+l,n),i] + UᵀW*S[i,1:p]
    UᵀW[:,:] += U[i,:] * (W[i,1:p])'
end

A₀ = tril(U*V',-1) + Matrix(B) + triu(W*S',1)
UᵀA_true = (U[j:end,:])'*A₀[j:end,j:end]

UᵀA - UᵀA_true

