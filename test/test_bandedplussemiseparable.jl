using SemiseparableMatrices, Test

@testset "BandedPlusSemiseparable" begin
    n = 20
    l, m, r, p = 4, 5, 2, 3
    B = brandn(n,n,l,m)
    U,V = randn(n,r), randn(n,r)
    W,S = randn(n,p), randn(n,p)
    A = BandedPlusSemiseparableMatrix(B, (U,V), (W,S))
    @test @inferred(size(A)) == (20,20)
    @test A ≈ B + tril(U*V',-1) + triu(W*S',1)

    @test_broken A[1:10,1:10] isa BandedPlusSemiseparableMatrix

    b = randn(n)
    @test ldiv!(UpperTriangular(A), copy(b)) ≈ UpperTriangular(Matrix(A)) \ b
end