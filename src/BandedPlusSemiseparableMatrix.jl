struct BandedPlusSemiseparableMatrix{T} <: LayoutMatrix{T}
    # representing B + tril(UV', -1) + triu(WS', 1)
    B::BandedMatrix{T, Matrix{T}, Base.OneTo{Int}}
    U::Matrix{T}
    V::Matrix{T}
    W::Matrix{T}
    S::Matrix{T}
end

function BandedPlusSemiseparableMatrix(B, (U,V), (W,S))
    if size(U,1) == size(V,1) == size(W,1) == size(S,1) == size(B,1) == size(B,2) && size(U,2) == size(V,2) && size(W,2) == size(S,2)
        BandedPlusSemiseparableMatrix(B, U, V, W, S)
    else
throw(DimensionMismatch("Dimensions are not compatible."))
    end
end

size(A::BandedPlusSemiseparableMatrix) = size(A.B)
copy(A::BandedPlusSemiseparableMatrix) = A # not mutable

function getindex(A::BandedPlusSemiseparableMatrix, k::Integer, j::Integer)
    if j > k
        view(A.W, k, :)' * view(A.S, j, :) + A.B[k,j]
    elseif k > j
        view(A.U, k, :)' * view(A.V, j, :) + A.B[k,j]
    else
        A.B[k,j]
    end
end

function ldiv!(R::UpperTriangular{<:Any,<:BandedPlusSemiseparableMatrix}, b::StridedVector)
    F = parent(R)
    n, p = size(F.W)
    l, m = bandwidths(F.B)
    T = eltype(F.S)
    sx = zeros(T, p)
    for j = n : -1 : 1
        residual = view(F.W, j, :)' * sx 
        for k = j+1 : min(j+m, n)
            residual += F.B[j, k] * b[k]
        end
        b[j] = (b[j] - residual) / F.B[j, j]
        #sx += F.S[j, :] * b[j]
        mul!(sx, I, view(F.S, j, :), b[j], one(T))
    end
end