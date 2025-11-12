using BandedMatrices, LinearAlgebra, Random, SemiseparableMatrices, Test
using SemiseparableMatrices: BandedPlusSemiseparableQRPerturbedFactors
#Random.seed!(1234)

@testset "QR" begin
    n = 20
    l, m, r, p = 4, 5, 2, 3
    B = brandn(n,n,l,m)

    @testset "BandedPlusSemiseparableQRPerturbedFactors" begin
        U,V = randn(n,r), randn(n,r)
        W,S = randn(n,p), randn(n,p)
        A = BandedPlusSemiseparableQRPerturbedFactors(U,V,W,S,B)
        fact_true = LinearAlgebra.qrfactUnblocked!(Matrix(A))
        fact = qr!(A)
        @test A ≈ fact_true.factors ≈ fact.factors
        @test fact_true.τ ≈ fact.τ
    end

    @testset "BandedPlusSemiseparableMatrix" begin
        U,V = randn(n,r), randn(n,r)
        W,S = randn(n,p), randn(n,p)
        A = BandedPlusSemiseparableMatrix(B,(U,V),(W,S))
        A_true = Matrix(A)
        fact_true = LinearAlgebra.qrfactUnblocked!(Matrix(A))
        fact = qr(A)
        @test A ≈ A_true
        @test fact_true.factors ≈ fact.factors
        @test fact_true.τ ≈ fact.τ
    end
end