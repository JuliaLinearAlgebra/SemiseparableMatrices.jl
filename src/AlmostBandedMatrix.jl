## AlmostBandedMatrix



struct AlmostBandedMatrix{T} <: AbstractMatrix{T}
    bands::BandedMatrix{T}
    fill::LowRankMatrix{T}
    function AlmostBandedMatrix{T}(bands::BandedMatrix{T}, fill::LowRankMatrix{T}) where T
        if size(bands) ≠ size(fill)
            error("Data and fill must be compatible size")
        end
        new{T}(bands,fill)
    end
end

AlmostBandedMatrix(bands::BandedMatrix, fill::LowRankMatrix) =
    AlmostBandedMatrix{promote_type(eltype(bands),eltype(fill))}(bands,fill)

AlmostBandedMatrix{T}(::UndefInitializer, nm::NTuple{2,Integer}, lu::NTuple{2,Integer}, r::Integer) where {T} =
    AlmostBandedMatrix(BandedMatrix{T}(undef,nm,lu), LowRankMatrix{T}(undef,nm,r))

function AlmostBandedMatrix{T}(A::AlmostBandedMatrix, (l,u)::NTuple{2,Integer}) where T
    B_in,L = bandpart(A),fillpart(A)
    l_in,u_in = bandwidths(B_in)
    B = BandedMatrix(B_in, (l,u))
    for b = u_in+1:u
        B[band(b)] .= L[band(b)]
    end
    AlmostBandedMatrix{T}(B, copy(L))
end

AlmostBandedMatrix(A::AlmostBandedMatrix, (l,u)::NTuple{2,Integer}) where T = AlmostBandedMatrix{T}(A, (l,u))



AlmostBandedMatrix{T}(Z::Zeros, lu::NTuple{2,Integer}, r::Integer) where {T} =
    AlmostBandedMatrix(BandedMatrix{T}(Z, lu), LowRankMatrix{T}(Z, r))

AlmostBandedMatrix(Z::AbstractMatrix, lu::NTuple{2,Integer}, r::Integer) =
    AlmostBandedMatrix{eltype(Z)}(Z, lu, r)

for MAT in (:AlmostBandedMatrix, :AbstractMatrix, :AbstractArray)
    @eval convert(::Type{$MAT{T}}, A::AlmostBandedMatrix) where {T} =
        AlmostBandedMatrix(AbstractMatrix{T}(A.bands),AbstractMatrix{T}(A.fill))
end


bandpart(A::AlmostBandedMatrix) = A.bands
fillpart(A::AlmostBandedMatrix) = A.fill

size(A::AlmostBandedMatrix) = size(A.bands)
Base.IndexStyle(::Type{ABM}) where {ABM<:AlmostBandedMatrix} =
    IndexCartesian()


function getindex(B::AlmostBandedMatrix,k::Integer,j::Integer)
    if j > k + bandwidth(B.bands,2)
        B.fill[k,j]
    else
        B.bands[k,j]
    end
end

# can only change the bands, not the fill
function setindex!(B::AlmostBandedMatrix, v, k::Integer, j::Integer)
        B.bands[k,j] = v
end

###
# QR
##

qr(A::AlmostBandedMatrix) = almostbanded_qr(A)
qr!(A::AlmostBandedMatrix) = almostbanded_qr!(A)

almostbanded_qr(A) = _almostbanded_qr(axes(A), A)
_almostbanded_qr(_, A) = qr!(AlmostBandedMatrix{float(eltype(A))}(A, (bandwidth(A,1),bandwidth(A,1)+bandwidth(A,2))))


function _almostbanded_qr!(A::AlmostBandedMatrix{T} , τ::AbstractVector{T}) where T
    B,L = bandpart(A),fillpart(A)
    l,u = bandwidths(B)

    m,n = size(A)
    k = 1
    while k ≤ min(m - 1 + !(T<:Real), n)
        kr = k:min(k+l+u,m)
        jr1 = k:min(k+u,n)
        jr2 = k+u+1:min(kr[end]+u,n)
        S = @view B[kr,jr1]
        τv = view(τ,jr1)

        R,_ = _banded_qr!(S, τv)
        Q = QRPackedQ(R,τv)

        B_right = @view B[kr,jr2] 
        L_right = @view L[kr,jr2]
        # The following writes it as Q' * (B -L) + Q'*L 
        # that is,  subtract out L from B, apply Q' to both
        # and add it back in again
        for j=1:size(B_right,2)
            B_right[j+1:end,j] -= L_right[j+1:end,j]
        end
        lmul!(Q', B_right)
        U,V = arguments(L_right)
        lmul!(Q', U)
        for j=1:size(B_right,2)
            B_right[j+1:end,j] += L_right[j+1:end,j]
        end
        k = jr1[end]+1
    end
    A, τ
end

function almostbanded_qr!(R::AbstractMatrix{T}, τ) where T 
    _almostbanded_qr!(R, τ)
    QR(R, τ)
 end
 
 almostbanded_qr!(R::AbstractMatrix{T}) where T = almostbanded_qr!(R, zeros(T, min(size(R)...)))
 


