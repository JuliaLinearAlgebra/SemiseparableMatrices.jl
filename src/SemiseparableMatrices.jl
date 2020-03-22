module SemiseparableMatrices
using ArrayLayouts, BandedMatrices, LazyArrays, LinearAlgebra, MatrixFactorizations, Base

import Base: size, getindex, setindex!, convert
import MatrixFactorizations: QR, QRPackedQ, getQ, getR
import LinearAlgebra: qr, qr!, lmul!, ldiv!, rmul!, triu!, factorize
import BandedMatrices: _banded_qr!
import LazyArrays: arguments
import ArrayLayouts: MemoryLayout, sublayout, sub_materialize, @lazyldiv, MatLdivVec, materialize!, triangularlayout, triangulardata


export SemiseparableMatrix, AlmostBandedMatrix, LowRankMatrix

const LowRankMatrix{T,A,B} = MulMatrix{T,Tuple{A,B}}
LowRankMatrix(A::AbstractArray, B::AbstractArray) = ApplyMatrix(*, A, B)
LowRankMatrix(S::SubArray) = LowRankMatrix(map(Array,arguments(S))...)

include("SemiseparableMatrix.jl")
include("AlmostBandedMatrix.jl")

end # module
