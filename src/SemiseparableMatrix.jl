# A semi-separable matrix is defined by
#
# S = triu(uvᵀ, bu+1) + tril(pqᵀ, -bl-1),
#
# where bu, bl >= 0. u and v are n × ru matrices and p and q are n × rl matrices.
# See [1] for the definition.

# [1] Chandrasekaran and Gu, Fast and stable algorithms for banded plus
# Semiseparable systems of linear equations, SIMAX, 25 (2003), pp. 373-384.

## Constructors

struct SemiseparableMatrix{T} <: AbstractMatrix{T}
    L::LowRankMatrix{T}
    U::LowRankMatrix{T}
    bl::Int
    bu::Int

    function SemiseparableMatrix(L::LowRankMatrix{T}, U::LowRankMatrix{T}, bl, bu) where T
        Lm, Ln = size(L)
        Um, Un = size(U)
        @assert Um == Un == Lm == Ln && Un >= bu+1 && Lm >= bl+1
        new{T}(L, U, bl, bu)
    end
end

function convert(::Type{Matrix}, S::SemiseparableMatrix)
    return triu(Matrix(S.U), S.bu+1) + tril(Matrix(S.L), -S.bl-1)
end

size(S::SemiseparableMatrix) = size(S.L)

function getindex(S::SemiseparableMatrix{T}, k::Int, j::Int)  where T
    k-j ≥ S.bl && return S.L[k,j]
    j-k ≥ S.bu && return S.U[k,j]
    return zero(T)
end
