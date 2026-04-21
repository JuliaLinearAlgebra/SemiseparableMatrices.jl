using BandedMatrices, LinearAlgebra, Random, SemiseparableMatrices, Test
using SemiseparableMatrices: SymmetricBPSPerturbedRQ
#Random.seed!(1234)

@testset "RQ" begin
    n = 20
    l = 4
    r = 5
    U,V = randn(n,r), randn(n,r)
    W = copy(V)
    S = copy(U)
    B = brandn(n,n,l,l)
    #make B symmetric
    for i in 1:n
        for j in 1:l
            B[i,min(i+j,n)] = B[min(i+j,n), i]
        end
    end

    A = BandedPlusSemiseparableMatrix(B,(U,V),(W,S))
    fact_true = LinearAlgebra.qrfactUnblocked!(Matrix(A))
    fact = qr(A)
    @test fact_true.factors ≈ fact.factors
    @test fact_true.τ ≈ fact.τ

    R = SymmetricBPSPerturbedRQ(fact.factors)
    τ = fact.τ
    SemiseparableMatrices.rq_mul!(R,τ)
    RQ = SemiseparableMatrices.rq_mul(fact.factors, τ)

    Q = fact_true.Q * I
    R₀ = fact_true.R
    RQ_true = R₀ * Q
    @test R ≈ RQ ≈ RQ_true
end