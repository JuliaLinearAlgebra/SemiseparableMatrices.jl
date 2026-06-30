using SemiseparableMatrices

# A benefit of our approach to QR is that it is in an LAPACK-compatible format. Thus for moderate n our
# O(n) factorization can either be combined with our O(n) solver, or LAPack's optimised O(n^2) solver.
# Here is a demonstration of using this. For small n the dense QR is only marginally slower:

n = 1000
l, m, r, p = 4, 5, 2, 3
B = brandn(n,n,l,m)
U,V = randn(n,r), randn(n,r)
W,S = randn(n,p), randn(n,p)
b = randn(n)

A = BandedPlusSemiseparableMatrix(B,(U,V),(W,S))
Ã = Matrix(A) # dense A
@time F = qr(A); # BPS QR, 0.01s
@time F̃ = qr(Ã); # Dense QRCompactWY, 0.04s
F̄ = QR(Matrix(F.factors), F.τ); # convert to dense format

@time F\b; # 0.02s
@time F̃\b; # 0.001s
@time F̄\b; # 0.004s

# But if we make n much bigger dense QR becomes prohibitively slow:

n = 10_000
l, m, r, p = 4, 5, 2, 3
B = brandn(n,n,l,m)
U,V = randn(n,r), randn(n,r)
W,S = randn(n,p), randn(n,p)
b = randn(n)

A = BandedPlusSemiseparableMatrix(B,(U,V),(W,S))
Ã = Matrix(A) # dense A
@time F = qr(A); # BPS QR, 0.11s
@time F̃ = qr(Ã); # Dense QRCompactWY, 27s
@time F̄ = QR(Matrix(F.factors), F.τ); # convert to dense format 2s, A 10x speedup!

@time F.R\(F.Q'b); # 0.001s
@time F̃\b; # 0.08s
@time F̄\b; # 0.25s