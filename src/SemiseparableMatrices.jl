module SemiseparableMatrices
using LinearAlgebra: BlasFloat
using ArrayLayouts, BandedMatrices, LazyArrays, LinearAlgebra, MatrixFactorizations, Base

import Base: size, getindex, setindex!, convert, copyto!, copy, axes
import MatrixFactorizations: QR, QRPackedQ, getQ, getR, QRPackedQLayout, AdjQRPackedQLayout
import LinearAlgebra: qr, qr!, lmul!, ldiv!, rmul!, triu!, factorize, rank
import BandedMatrices: _banded_qr!, bandeddata, resize
import LazyArrays: arguments, applylayout, _cache, CachedArray, CachedMatrix, ApplyLayout, resizedata!, PaddedRows
import ArrayLayouts: MemoryLayout, sublayout, sub_materialize, MatLdivVec, materialize!, triangularlayout, 
                        triangulardata, zero!, _copyto!, colsupport, rowsupport,
                        _qr, _qr!, _factorize

export SemiseparableMatrix, AlmostBandedMatrix, LowRankMatrix, ApplyMatrix, ApplyArray, almostbandwidths, almostbandedrank, BandedPlusSemiseparableMatrix

LazyArraysBandedMatricesExt = Base.get_extension(LazyArrays, :LazyArraysBandedMatricesExt)
BandedLayouts = LazyArraysBandedMatricesExt.BandedLayouts
ApplyBandedLayout = LazyArraysBandedMatricesExt.ApplyBandedLayout

const LowRankMatrix{T,A,B} = MulMatrix{T,Tuple{A,B}}
LowRankMatrix(A::AbstractArray, B::AbstractArray) = ApplyMatrix(*, A, B)
LowRankMatrix(S::SubArray) = LowRankMatrix(map(Array,arguments(S))...)
LowRankMatrix{T}(::UndefInitializer, (m,n)::NTuple{2,Integer}, r::Integer) where T = 
    ApplyMatrix(*, Array{T}(undef, m, r), Array{T}(undef, r, n))
LowRankMatrix(Z::Zeros{T}, r::Integer) where T = ApplyMatrix(*, zeros(T, size(Z,1), r), zeros(T, r, size(Z,2)))
LowRankMatrix{T}(Z::Zeros, r::Integer) where T = ApplyMatrix(*, zeros(T, size(Z,1), r), zeros(T, r, size(Z,2)))

separablerank(A) = size(arguments(ApplyLayout{typeof(*)}(),A)[1],2)    

include("SemiseparableMatrix.jl")
include("AlmostBandedMatrix.jl")
include("invbanded.jl")
include("BandedPlusSemiseparableMatrices.jl")

end # module
