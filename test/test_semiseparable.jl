using SemiseparableMatrices, BandedMatrices, LinearAlgebra, Test
import SemiseparableMatrices: LowRankMatrix

@testset "LowRankMatrix" begin
    @test LowRankMatrix(Zeros(5,5), 2) == Zeros(5,5)
end


@testset "SemiseparableMatrix" begin
    n = 10
    v = rand(n, 2); w = rand(n, 2); y = rand(n, 3); z = rand(n, 3);
    U = ApplyMatrix(*, v, w'); L = ApplyMatrix(*, y, z')
    @test SemiseparableMatrix(L, U, 1, 2) isa SemiseparableMatrix{Float64}
    @test Matrix(SemiseparableMatrix(L, U, 1, 2)) ≈ tril(L, -1) + triu(U, 2)
    @test_throws ArgumentError SemiseparableMatrix(L, U, 11, 2)
    @test_throws ArgumentError SemiseparableMatrix(L, U, 1, 11)

    y = rand(n+1, 3); z = rand(n+1, 3)
    U = LowRankMatrix(v, w'); L = LowRankMatrix(y, z')
    @test_throws DimensionMismatch SemiseparableMatrix(L, U, 1, 2)
end
