# A semi-separable matrix is defined by
#
# S = triu(uvᵀ, bu+1) + tril(pqᵀ, -bl-1),
#
# where bu, bl >= 0. u and v are n × ru matrices and p and q are n × rl matrices.
# See [1] for the definition.

# [1] Chandrasekaran and Gu, Fast and stable algorithms for banded plus
# Semiseparable systems of linear equations, SIMAX, 25 (2003), pp. 373-384.

## Constructors

struct SemiseparableMatrix{T,LL<:AbstractMatrix{T},UU<:AbstractMatrix{T}} <: AbstractMatrix{T}
    L::LL
    U::UU
    bl::Int
    bu::Int

    function SemiseparableMatrix{T,LL,UU}(L::LL, U::UU, bl::Integer, bu::Integer) where {T,LL<:AbstractMatrix{T},UU<:AbstractMatrix{T}}
        Lm, Ln = size(L)
        Um, Un = size(U)
        if !(Um == Un == Lm == Ln && Un >= bu+1 && Lm >= bl+1)
            throw(ArgumentError())
        end
        new{T,LL,UU}(L, U, bl, bu)
    end
end

SemiseparableMatrix(A::LL, B::UU, bl::Integer, bu::Integer) where {T,LL<:AbstractMatrix{T},UU<:AbstractMatrix{T}} = 
    SemiseparableMatrix{T,LL,UU}(A, B, bl, bu)


function convert(::Type{Matrix}, S::SemiseparableMatrix)
    return triu(Matrix(S.U), S.bu+1) + tril(Matrix(S.L), -S.bl-1)
end

size(S::SemiseparableMatrix) = size(S.L)

function getindex(S::SemiseparableMatrix{T}, k::Int, j::Int)  where T
    k-j ≥ S.bl && return S.L[k,j]
    j-k ≥ S.bu && return S.U[k,j]
    return zero(T)
end
