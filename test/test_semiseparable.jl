using SemiseparableMatrices, LowRankApprox, BandedMatrices, LinearAlgebra, Test

# Constructions:
@testset "construction" begin
    let n = 10, v = rand(n, 2), w = rand(n, 2), y = rand(n, 3), z = rand(n, 3), U = LowRankMatrix(v, w), L = LowRankMatrix(y, z)
        @test SemiseparableMatrix(U, L, 1, 2) isa SemiseparableMatrix{Float64}
        @test Matrix(SemiseparableMatrix(U, L, 1, 2)) == triu(U, 2)+tril(L, -3)
        @test_throws AssertionError SemiseparableMatrix(U, L, 10, 2)
        @test_throws AssertionError SemiseparableMatrix(U, L, 1, 10)
    end

    let v = rand(n, 2), w = rand(n, 2), y = rand(n+1, 3), z = rand(n+1, 3), U = LowRankMatrix(v, w), L = LowRankMatrix(y, z)
        @test_throws AssertionError SemiseparableMatrix(U, L, 1, 2)
    end
end
