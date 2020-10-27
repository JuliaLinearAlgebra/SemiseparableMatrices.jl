# A semi-separable matrix is defined by
#
# S = triu(uvᵀ, u) + tril(pqᵀ, -bl),
#
# where u, bl >= 0. u and v are n × ru matrices and p and q are n × rl matrices.
# See [1] for the definition.

# [1] Chandrasekaran and Gu, Fast and stable algorithms for banded plus
# Semiseparable systems of linear equations, SIMAX, 25 (2003), pp. 373-384.

## Constructors

struct SemiseparableMatrix{T,LL<:AbstractMatrix{T},UU<:AbstractMatrix{T}} <: LayoutMatrix{T}
    L::LL
    U::UU
    l::Int
    u::Int

    function SemiseparableMatrix{T,LL,UU}(L::LL, U::UU, l::Integer, u::Integer) where {T,LL<:AbstractMatrix{T},UU<:AbstractMatrix{T}}
        Lm, Ln = size(L)
        Um, Un = size(U)
        Um == Un == Lm == Ln || throw(DimensionMismatch())
        (Un >= u && Lm >= l) || throw(ArgumentError("l and u should be compatible with dimensions"))
        -l < u || throw(ArgumentError("bands must not overlap"))
        new{T,LL,UU}(L, U, l, u)
    end
end

SemiseparableMatrix(A::LL, B::UU, l::Integer, u::Integer) where {T,LL<:AbstractMatrix{T},UU<:AbstractMatrix{T}} = 
    SemiseparableMatrix{T,LL,UU}(A, B, l, u)


function convert(::Type{Matrix}, S::SemiseparableMatrix)
    return triu(Matrix(S.U), S.u) + tril(Matrix(S.L), -S.l)
end

size(S::SemiseparableMatrix) = size(S.L)

function getindex(S::SemiseparableMatrix{T}, k::Int, j::Int)  where T
    k-j ≥ S.l && return S.L[k,j]
    j-k ≥ S.u && return S.U[k,j]
    return zero(T)
end


# triangular

colsupport(L::SemiseparableMatrix{<:Any,<:Any,<:Zeros}, j) = minimum(j)+L.l:size(L,1)
colsupport(U::SemiseparableMatrix{<:Any,<:Zeros}, j) = 1:maximum(j)-U.u
rowsupport(L::SemiseparableMatrix{<:Any,<:Any,<:Zeros}, k) = 1:maximum(k)-L.l
rowsupport(U::SemiseparableMatrix{<:Any,<:Zeros}, k) = minimum(k)+U.u:size(U,2)
