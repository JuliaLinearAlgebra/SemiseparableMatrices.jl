using SemiseparableMatrices, BandedMatrices, FillArrays, Test, LinearAlgebra, MatrixFactorizations, LazyArrays, ArrayLayouts
import BandedMatrices: _BandedMatrix, _banded_qr!
import SemiseparableMatrices: bandpart, fillpart, AlmostBandedLayout
import MatrixFactorizations: QRPackedQ

@testset "AlmostBandedMatrix" begin
    @testset "Slices" begin
        n = 80
        A = AlmostBandedMatrix(BandedMatrix(Fill(2.0,n,n),(1,1)), LowRankMatrix(fill(3.0,n), ones(1,n)))
        V = view(A,1:3,1:3)
        @test MemoryLayout(typeof(A)) == MemoryLayout(typeof(V)) == AlmostBandedLayout()
        @test AlmostBandedMatrix(V) == A[1:3,1:3] == V
        @test A[1:3,1:3] isa AlmostBandedMatrix 
    end

    @testset "QR" begin
        n = 80
        A = AlmostBandedMatrix(BandedMatrix(Fill(2.0,n,n),(1,1)), LowRankMatrix(fill(3.0,n), ones(1,n)))
        Ã = deepcopy(A)
        B,L = bandpart(A),fillpart(A)
        Ā = AlmostBandedMatrix(A,(1,2))
        @test A == Ā == B + triu(Matrix(L),2)
        F = qr(A)
        @test F.Q' * A ≈ F.R
        @test A == Ã

        Ā = AlmostBandedMatrix(A,(1,2))
        τ = Vector{Float64}(undef,n)
        @inferred(SemiseparableMatrices._almostbanded_qr!(Ā,τ))

        @test F.Q isa QRPackedQ{Float64,<:BandedMatrix}
        F.R
    end
end

