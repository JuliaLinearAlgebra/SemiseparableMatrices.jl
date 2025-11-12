using SemiseparableMatrices, Test

@testset "BandedPlusSemiseparable" begin
    n = 20
    l, m, r, p = 4, 5, 2, 3
    B = brandn(n,n,l,m)
    U,V = randn(n,r), randn(n,r)
    W,S = randn(n,p), randn(n,p)
    @test BandedPlusSemiseparableMatrix(B, (U,V), (W,S)) ≈ B + tril(U*V',-1) + triu(W*S',1)
end