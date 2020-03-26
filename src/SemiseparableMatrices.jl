module SemiseparableMatrices
using ArrayLayouts, BandedMatrices, LazyArrays, LinearAlgebra, MatrixFactorizations, Base

import Base: size, getindex, setindex!, convert, copyto!
import MatrixFactorizations: QR, QRPackedQ, getQ, getR
import LinearAlgebra: qr, qr!, lmul!, ldiv!, rmul!, triu!, factorize, rank
import BandedMatrices: _banded_qr!, bandeddata
import LazyArrays: arguments, applylayout, _cache, CachedArray, ApplyLayout
import ArrayLayouts: MemoryLayout, sublayout, sub_materialize, MatLdivVec, materialize!, triangularlayout, triangulardata, zero!

export SemiseparableMatrix, AlmostBandedMatrix, LowRankMatrix, ApplyMatrix, ApplyArray, almostbandwidths, almostbandedrank

const LowRankMatrix{T,A,B} = MulMatrix{T,Tuple{A,B}}
LowRankMatrix(A::AbstractArray, B::AbstractArray) = ApplyMatrix(*, A, B)
LowRankMatrix(S::SubArray) = LowRankMatrix(map(Array,arguments(S))...)
LowRankMatrix{T}(::UndefInitializer, (m,n)::NTuple{2,Integer}, r::Integer) where T = 
    ApplyMatrix(*, Array{T}(undef, m, r), Array{T}(undef, r, n))

separablerank(A::LowRankMatrix) = size(A.args[1],2)    

include("SemiseparableMatrix.jl")
include("AlmostBandedMatrix.jl")

end # module
