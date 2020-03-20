using SemiseparableMatrices, BandedMatrices, FillArrays, Test, LinearAlgebra, MatrixFactorizations, LazyArrays
import BandedMatrices: _BandedMatrix, _banded_qr!
import SemiseparableMatrices: bandpart, fillpart
import MatrixFactorizations: QRPackedQ
import LazyArrays: arguments

# Constructions:
@testset "construction" begin
    n = 10
    A = AlmostBandedMatrix(BandedMatrix(Fill(2.0,n,n),(1,1)), LowRankMatrix(fill(3.0,n), ones(1,n)))
    Ã = deepcopy(A)
    B,L = bandpart(A),fillpart(A)
    Ā = AlmostBandedMatrix(A,(1,2))
    @test A == Ā == B + triu(Matrix(L),2)
    F = qr(A)
    @test A == Ã

    @test F.Q' * A ≈ F.R
end
