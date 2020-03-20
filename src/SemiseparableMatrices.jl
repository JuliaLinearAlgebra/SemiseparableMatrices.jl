module SemiseparableMatrices
using ArrayLayouts, BandedMatrices, LazyArrays, LinearAlgebra, MatrixFactorizations, Base

import Base: size, getindex, convert
import MatrixFactorizations: QR, QRPackedQ
import LinearAlgebra: qr, qr!
import BandedMatrices: _banded_qr!
import LazyArrays: arguments

export SemiseparableMatrix, AlmostBandedMatrix, LowRankMatrix

const LowRankMatrix{T,A,B} = MulMatrix{T,Tuple{A,B}}
LowRankMatrix(A::AbstractArray, B::AbstractArray) = ApplyMatrix(*, A, B)

include("SemiseparableMatrix.jl")
include("AlmostBandedMatrix.jl")

end # module
