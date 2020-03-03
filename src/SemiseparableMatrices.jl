module SemiseparableMatrices
using ArrayLayouts, BandedMatrices, LowRankApprox, LinearAlgebra, Base

import Base: size, getindex, convert

export SemiseparableMatrix, AlmostBandedMatrix

include("SemiseparableMatrix.jl")

end # module
