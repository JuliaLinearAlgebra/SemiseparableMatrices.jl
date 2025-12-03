using BandedMatrices, LinearAlgebra, Random, SemiseparableMatrices, Test
using SemiseparableMatrices: BandedPlusSemiseparableQRPerturbedFactors
Random.seed!(1234)

@testset "QR" begin
    n = 20
    l, m, r, p = 4, 5, 2, 3
    B = brandn(n,n,l,m)

    @testset "BandedPlusSemiseparableQRPerturbedFactors" begin
        U,V = randn(n,r), randn(n,r)
        W,S = randn(n,p), randn(n,p)
        A = BandedPlusSemiseparableQRPerturbedFactors(B, (U,V), (W,S))
        @test @inferred(size(A)) == (20,20)
        fact_true = LinearAlgebra.qrfactUnblocked!(Matrix(A))
        fact = @inferred(qr!(A))
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

        b = randn(n)
        Q,R = fact

        @test R ≡ UpperTriangular(fact.factors)

        res = lmul!(Q',b)
        println(res)
        F = fact.factors
        τ = fact.τ
        for i = 1 : n-1
            y = zeros(n)
            y[i] = 1
            y[i+1:n] = F[i+1:n, i]
            b = (I-τ[i]*y*y')*b
        end
        @test b ≈ res
        #x = R \ b
        #ldiv!(R, b)
        #@test x ≈ b
    end
end