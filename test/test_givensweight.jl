using SemiseparableMatrices, BandedMatrices, LazyArrays, LinearAlgebra, Test
import SemiseparableMatrices: GivensWeightMatrix

@testset "Givens-weight inverse of upper banded" begin
    @testset "n = $n, m = $m" for n in (1, 10, 100, 200), m in (1, 2, 4, 8)
        B = brand(n, n, 0, m)
        B[band(0)] .+= m + 1
        @test GivensWeightMatrix(ApplyArray(inv, B)) ≈ inv(Matrix(B))
    end
end