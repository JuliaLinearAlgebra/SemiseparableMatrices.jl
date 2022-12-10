using Random: BigFloat
using SemiseparableMatrices, BandedMatrices, Test, LinearAlgebra, MatrixFactorizations, LazyArrays, ArrayLayouts, Random
import BandedMatrices: _BandedMatrix, _banded_qr!
import SemiseparableMatrices: bandpart, fillpart, AlmostBandedLayout, VcatAlmostBandedLayout, resizedata!, _almostbanded_qr!
import MatrixFactorizations: QRPackedQ
import LazyArrays: CachedArray

Random.seed!(0)

@testset "AlmostBandedMatrix" begin
    @testset "Constructors" begin
        A = AlmostBandedMatrix{Float64}(undef, (10,11), (2,1), 2)
        A[1,1] = 2
        @test A[1,1] == 2
        A[4,1] = 0
        @test A[4,1] == 0.0
        @test_throws BandError A[4,1] = 2
        if VERSION < v"1.8-"
            @test_throws ErrorException A[1,3] = 5
        else
            @test_throws Base.CanonicalIndexError A[1,3] = 5
        end
        @test almostbandwidths(A) == (2,1)
        @test almostbandedrank(A) == 2
    end
    
    @testset "copyto!" begin
        n = 10
        A = Vcat(Ones(1,n), BandedMatrix((0 => -Ones(n-1), 1 => 1:(n-1)), (n-1,n)))
        @test MemoryLayout(A) == VcatAlmostBandedLayout()
        @test almostbandwidths(A) == (1,0)
        @test almostbandedrank(A) == 1

        dest = AlmostBandedMatrix{Float64}(undef, size(A), almostbandwidths(A), almostbandedrank(A))
        copyto!(dest, A)
        @test dest == A 

        dest = AlmostBandedMatrix{Float64}(undef, (1,2), (1,1), 1)
        copyto!(dest, view(A,1:1,1:2))
        @test dest == A[1:1,1:2]

        @test AlmostBandedMatrix(A) == Matrix(A)

        V = view(A,1:5,2:5)
        @test MemoryLayout(V) == VcatAlmostBandedLayout()
        @test almostbandwidths(V) == (2,-1)
        @test almostbandedrank(V) == 1
        @test A[1:5,2:5] isa AlmostBandedMatrix

        @test AlmostBandedMatrix(V) == Matrix(V) == V == A[1:5,2:5]

        

        V = view(A,Base.OneTo(5),Base.OneTo(5))
        @test AlmostBandedMatrix(V) == V == Matrix(V)

        V = view(AlmostBandedMatrix(A),1:5,2:5)
        @test MemoryLayout(V) == AlmostBandedLayout()
        @test AlmostBandedMatrix(V) == V == Matrix(V)

        A = Vcat(randn(2,n), BandedMatrix((0 => -Ones(n-1), 1 => 1:(n-1), 2 => Ones(n-2)), (n-2,n)))
        @test MemoryLayout(A) isa VcatAlmostBandedLayout
        @test almostbandedrank(A) == 2
        @test almostbandwidths(A) == (2,0)
        @test AlmostBandedMatrix(A) == Matrix(A)

        A = Vcat(randn(1,n), randn(1,n), BandedMatrix((0 => -Ones(n-1), 1 => 1:(n-1), 2 => Ones(n-2)), (n-2,n)))
        @test MemoryLayout(A) isa VcatAlmostBandedLayout
        @test almostbandedrank(A) == 2
        @test almostbandwidths(A) == (2,0)
        @test AlmostBandedMatrix(A) == Matrix(A)
    end

    @testset "cache" begin
        n = 10
        A = Vcat(Ones(1,n), BandedMatrix((0 => -Ones(n-1), 1 => 1:(n-1)), (n-1,n)))
        C = cache(A); resizedata!(C,2,2); 
        @test C.data[1:2,1:2] == [1.0 1.0; -1.0 1.0]
        C = cache(A);
        resizedata!(C,10,10);
        @test C.data[Base.OneTo.(C.datasize)...] == A[1:10,1:10]
        @test C isa LazyArrays.CachedMatrix{Float64,<:AlmostBandedMatrix}

        l,u = almostbandwidths(A)
        r = almostbandedrank(A)
        data = AlmostBandedMatrix{Float64}(undef,(2l+u+1,0),(l,l+u),r) # pad super
        C = CachedArray(data,A);  resizedata!(C,1,2); 
        @test C.data[1,2] == 1.0
        resizedata!(C,5,3); 
        @test C.data[1:5,1:3] == A[1:5,1:3]
        

        A = Vcat(randn(2,n), BandedMatrix((0 => -Ones(n-1), 1 => 1:(n-1), 2 => Ones(n-2)), (n-2,n)))
        C = cache(A);
        resizedata!(C,10,10);
        @test C.data[Base.OneTo.(C.datasize)...] == A[1:10,1:10]
        @test C isa LazyArrays.CachedMatrix{Float64,<:AlmostBandedMatrix}

        @testset "BigFloat" begin
            A = Vcat(randn(2,n), BandedMatrix{BigFloat}((0 => -Ones(n-1), 1 => 1:(n-1), 2 => Ones(n-2)), (n-2,n)))
            C = cache(A);
            resizedata!(C,10,10);
            @test C.data[Base.OneTo.(C.datasize)...] == A[1:10,1:10]
            @test C isa LazyArrays.CachedMatrix{BigFloat,<:AlmostBandedMatrix}
        end
    end

    @testset "Slices" begin
        n = 80
        A = AlmostBandedMatrix(BandedMatrix(fill(2.0,n,n),(1,1)), LowRankMatrix(fill(3.0,n), ones(1,n)))
        V = view(A,1:3,1:3)
        @test MemoryLayout(A) == MemoryLayout(V) == AlmostBandedLayout()
        @test AlmostBandedMatrix(V) == A[1:3,1:3] == V
        @test A[1:3,1:3] isa AlmostBandedMatrix 
    end

    @testset "colsupport" begin
        A = AlmostBandedMatrix{Float64}(undef, (10,11), (2,1), 2)
        @test colsupport(A,3) ≡ Base.OneTo(5)
        @test rowsupport(A,5) ≡ 3:11
    end

    @testset "Triangular" begin
        n = 80
        A = AlmostBandedMatrix(BandedMatrix(fill(2.0,n,n),(1,1)), LowRankMatrix(fill(3.0,n), ones(1,n))) 
        b = randn(n)
        @test MemoryLayout(UpperTriangular(A)) == TriangularLayout{'U','N',AlmostBandedLayout}()
        @test UpperTriangular(Matrix(A)) \ b ≈ UpperTriangular(A) \ b
        @test UnitUpperTriangular(Matrix(A)) \ b ≈ UnitUpperTriangular(A) \ b
        @test LowerTriangular(Matrix(A)) \ b ≈ LowerTriangular(A) \ b
        @test UnitLowerTriangular(Matrix(A)) \ b ≈ UnitLowerTriangular(A) \ b
    end

    @testset "QR" begin
        n = 80
        A = AlmostBandedMatrix(BandedMatrix(fill(2.0,n,n),(1,1)), LowRankMatrix(fill(3.0,n), ones(1,n)))
        A[band(0)] .+= 1:n
        Ã = deepcopy(A)
        B,L = bandpart(A),fillpart(A)
        Ā = AlmostBandedMatrix(A,(1,2))
        @test A == Ā == B + triu(Matrix(L),2)
        F = qr(A)
        @test F.Q isa LinearAlgebra.QRPackedQ{Float64,<:BandedMatrix}
        @test F.R isa UpperTriangular{Float64,<:SubArray{Float64,2,<:AlmostBandedMatrix}}
        @test F.Q' * A ≈ F.R
        @test A == Ã

        Ā = AlmostBandedMatrix(A,(1,2))
        τ = Vector{Float64}(undef,n)
        @inferred(SemiseparableMatrices._almostbanded_qr!(Ā,τ))

        b = randn(n)
        @test A \ b ≈ Matrix(A) \ b 
        @test all(A \ b .=== F \ b .=== F.R \ (F.Q'*b)) 
        Q̃ = QRPackedQ(F.factors,F.τ)
        @test Matrix(Q̃) ≈ Matrix(F.Q)
        @test lmul!(Q̃,copy(b)) ≈ lmul!(F.Q,copy(b)) ≈ Matrix(F.Q)*b
        @test lmul!(Q̃',copy(b)) ≈ lmul!(F.Q',copy(b)) ≈ Matrix(F.Q)'*b

        A = Vcat(randn(2,n), BandedMatrix((0 => -Ones(n-1), 1 => 1:(n-1), 2 => Ones(n-2)), (n-2,n)))
        @test MemoryLayout(A) isa VcatAlmostBandedLayout
        @test qr(A) isa MatrixFactorizations.QR{Float64,<:AlmostBandedMatrix}
        @test qr(A) \ b ≈ Matrix(A) \ b

        A = Vcat(randn(1,n), randn(1,n), BandedMatrix((0 => -Ones(n-1), 1 => 1:(n-1), 2 => Ones(n-2)), (n-2,n)))
        @test MemoryLayout(A) isa VcatAlmostBandedLayout
    end

    @testset "one-col qr" begin
        A = AlmostBandedMatrix{Float64}(undef,(2,2),(1,1),1)
        A.bands.data .= randn.()
        A.fill.args[1] .= randn.()
        A.fill.args[2] .= randn.()
        Ã = deepcopy(A)
        _almostbanded_qr!(A,[1.0],1)
        @test qr(Ã[:,1]).Q' * Ã ≈ UpperTriangular(A)
    end

    @testset "almost banded degenerate band" begin
        A = Vcat(randn(2,6), BandedMatrix(2 => 1:5)[1:4,:])
        @test almostbandwidths(A) == (1,0)
        @test colsupport(A,1) == Base.OneTo(2)
        b = randn(6)
        @test A\b ≈ Matrix(A)\b
    end
end

