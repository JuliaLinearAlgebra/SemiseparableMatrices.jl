## AlmostBandedMatrix

abstract type AbstractAlmostBandedLayout <: MemoryLayout end
struct AlmostBandedLayout <: AbstractAlmostBandedLayout end


struct AlmostBandedMatrix{T,D,A,B,R} <: LayoutMatrix{T}
    bands::BandedMatrix{T,D,R}
    fill::LowRankMatrix{T,A,B}
    AlmostBandedMatrix{T,D,A,B,R}(bands, fill) where {T,D,A,B,R} = new{T,D,A,B,R}(bands,fill)
end

function AlmostBandedMatrix{T}(bands::BandedMatrix{T,D,R}, fill::LowRankMatrix{T,A,B}) where {T,D,A,B,R}
    if size(bands) ≠ size(fill)
        error("Data and fill must be compatible size")
    end
    AlmostBandedMatrix{T,D,A,B,R}(bands,fill)
end

AlmostBandedMatrix(bands::BandedMatrix, fill::LowRankMatrix) =
    AlmostBandedMatrix{promote_type(eltype(bands),eltype(fill))}(bands,fill)

AlmostBandedMatrix(bands::AbstractMatrix, fill::AbstractMatrix) = 
    AlmostBandedMatrix(BandedMatrix(bands), LowRankMatrix(fill))

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

AlmostBandedMatrix(A::AlmostBandedMatrix{T}, (l,u)::NTuple{2,Integer}) where T = AlmostBandedMatrix{T}(A, (l,u))



AlmostBandedMatrix{T}(Z::Zeros, lu::NTuple{2,Integer}, r::Integer) where {T} =
    AlmostBandedMatrix(BandedMatrix{T}(Z, lu), LowRankMatrix{T}(Z, r))


AlmostBandedMatrix{T}(A::AbstractMatrix, lu::NTuple{2,Integer}=almostbandwidths(A), r::Integer=almostbandedrank(A)) where T = 
    copyto!(AlmostBandedMatrix{T}(undef, size(A), lu, r), A)

AlmostBandedMatrix(A::AbstractMatrix{T}, lu::NTuple{2,Integer}=almostbandwidths(A), r::Integer=almostbandedrank(A)) where T =
    AlmostBandedMatrix{T}(A, lu, r)

MemoryLayout(::Type{<:AlmostBandedMatrix}) = AlmostBandedLayout()

for MAT in (:AlmostBandedMatrix, :AbstractMatrix, :AbstractArray)
    @eval convert(::Type{$MAT{T}}, A::AlmostBandedMatrix) where {T} =
        AlmostBandedMatrix(AbstractMatrix{T}(A.bands),AbstractMatrix{T}(A.fill))
end


bandpart(A::AlmostBandedMatrix) = A.bands
fillpart(A::AlmostBandedMatrix) = A.fill

almostbandwidths(_, A) = bandwidths(bandpart(A))    
almostbandedrank(_, A) = separablerank(fillpart(A))
almostbandwidths(A) = almostbandwidths(MemoryLayout(typeof(A)), A)
almostbandedrank(A) = almostbandedrank(MemoryLayout(typeof(A)), A)

size(A::AlmostBandedMatrix) = size(A.bands)
Base.IndexStyle(::Type{ABM}) where {ABM<:AlmostBandedMatrix} =
    IndexCartesian()

function colsupport(::AbstractAlmostBandedLayout, A, j) 
    l,_ = almostbandwidths(A)
    Base.OneTo(min(maximum(j)+l,size(A,1)))
end

function rowsupport(::AbstractAlmostBandedLayout, A, k) 
    l,_ = almostbandwidths(A)
    max(1,minimum(k)-l):size(A,2)
end

function getindex(B::AlmostBandedMatrix, k::Integer, j::Integer)
    if j > k + bandwidth(B.bands,2)
        B.fill[k,j]
    else
        B.bands[k,j]
    end
end

# can only change the bands, not the fill
function setindex!(B::AlmostBandedMatrix, v, k::Integer, j::Integer)
    l,u = bandwidths(bandpart(B))
    if j-k ≤ u
        B.bands[k,j] = v
    else
        B.fill[k,j] = v
    end
    B
end

function triu!(A::AlmostBandedMatrix) 
    triu!(bandpart(A))
    A
end


###
# SubArray
###

sublayout(::AlmostBandedLayout, ::Type{<:Tuple{AbstractUnitRange{Int},AbstractUnitRange{Int}}}) = 
    AlmostBandedLayout()

sub_materialize(::AbstractAlmostBandedLayout, V) = AlmostBandedMatrix(V)

bandpart(V::SubArray) = view(bandpart(parent(V)), parentindices(V)...)
fillpart(V::SubArray) = view(fillpart(parent(V)), parentindices(V)...)

###
# QR
##

_factorize(::AbstractAlmostBandedLayout, _, A) = qr(A)
_qr(::AbstractAlmostBandedLayout, _, A) = almostbanded_qr(A)
_qr!(::AlmostBandedLayout, _, A) = almostbanded_qr!(A)

almostbanded_qr(A) = _almostbanded_qr(axes(A), A)
function _almostbanded_qr(_, A) 
    l,u = almostbandwidths(A)
    qr!(AlmostBandedMatrix{float(eltype(A))}(A, (l,l+u)))
end

function _almostbanded_qr!(A::AbstractMatrix{T}, τ::AbstractVector{T}) where T
    m,n = size(A)
    _almostbanded_qr!(A, τ, min(m - 1 + !(T<:Real), n))
end
function _almostbanded_qr!(A::AbstractMatrix , τ::AbstractVector, ncols)
    B,L = bandpart(A),fillpart(A)
    l,u = bandwidths(B)
    m,n = size(A)
    k = 1
    while k ≤ ncols
        kr = k:min(k+l+u,m)
        jr1 = k:min(k+u,n)
        jr2 = k+u+1:min(kr[end]+u,n)
        jr3 = k:min(k+u,n,ncols)
        S = view(B,kr,jr1)
        τv = view(τ,jr3)

        R,_ = _banded_qr!(S, τv, length(jr3))
        Q = QRPackedQ(R,τv)

        B_right = view(B,kr,jr2)
        L_right = view(L,kr,jr2)
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

getQ(F::QR{<:Any,<:AlmostBandedMatrix}) = LinearAlgebra.QRPackedQ(bandpart(F.factors), F.τ)
function getR(F::QR{<:Any,<:AlmostBandedMatrix}) 
    n = min(size(F.factors,1),size(F.factors,2))
    UpperTriangular(view(F.factors,1:n,1:n))
end

function ldiv!(A::QR{<:Any,<:AlmostBandedMatrix}, B::AbstractVector)
    R = A.factors
    lmul!(adjoint(A.Q), B)
    B .= Ldiv(UpperTriangular(R), B)
    B
end

function ldiv!(A::QR{<:Any,<:AlmostBandedMatrix}, B::AbstractMatrix)
    R = A.factors
    lmul!(adjoint(A.Q), B)
    B .= Ldiv(UpperTriangular(R), B)
    B
end

# needed for adaptive QR
materialize!(M::Lmul{<:QRPackedQLayout{AlmostBandedLayout}}) =
    lmul!(QRPackedQ(bandpart(M.A.factors), M.A.τ), M.B)
function materialize!(M::Lmul{<:AdjQRPackedQLayout{AlmostBandedLayout}})
    Q = M.A'
    lmul!(QRPackedQ(bandpart(Q.factors), Q.τ)', M.B)    
end

###
# UpperTriangular ldiv!
##

triangularlayout(::Type{Tri}, ::ML) where {Tri,ML<:AlmostBandedLayout} = Tri{ML}()

function _almostbanded_upper_ldiv!(Tri, R::AbstractMatrix, b::AbstractVector{T}, buffer) where T
    B = bandpart(R)
    L = fillpart(R)
    U,V = arguments(L)
    fill!(buffer, zero(T))

    l,u = bandwidths(B)
    k = n = size(R,2)

    while k > 0
        kr = max(1,k-u):k
        jr1 = k+1:k+u+1
        jr2 = k+u+2:k+2u+2
        bv = view(b,kr)
        if jr2[1] < n
            muladd!(one(T),view(V,:,jr2),view(b,jr2),one(T),buffer)
            muladd!(-one(T),view(U,kr,:),buffer,one(T),bv)
        end
        if jr1[1] < n
            muladd!(-one(T),view(R,kr,jr1),view(b,jr1),one(T),bv)
        end
        materialize!(Ldiv(Tri(view(B,kr,kr)), bv))
        k = kr[1]-1
    end
    b
end

@inline function materialize!(M::MatLdivVec{TriangularLayout{'U','N',AlmostBandedLayout}})
    R,x = M.A,M.B
    A = triangulardata(R)
    r = size(arguments(fillpart(A))[1],2)
    _almostbanded_upper_ldiv!(UpperTriangular, A, x, Vector{eltype(M)}(undef, r))
    x
end

@inline function materialize!(M::MatLdivVec{TriangularLayout{'U','U',AlmostBandedLayout}})
    R,x = M.A,M.B
    A = triangulardata(R)
    r = size(arguments(fillpart(A))[1],2)
    _almostbanded_upper_ldiv!(UnitUpperTriangular, A, x, Vector{eltype(M)}(undef, r))
    x
end

@inline function materialize!(M::MatLdivVec{TriangularLayout{'L','N',AlmostBandedLayout}})
    R,x = M.A,M.B
    A = triangulardata(R)
    materialize!(Ldiv(LowerTriangular(bandpart(A)),x))
    x
end

@inline function materialize!(M::MatLdivVec{TriangularLayout{'L','U',AlmostBandedLayout}})
    R,x = M.A,M.B
    A = triangulardata(R)
    materialize!(Ldiv(UnitLowerTriangular(bandpart(A)),x))
    x
end


###
# VcatBanded
###

struct VcatAlmostBandedLayout <: AbstractAlmostBandedLayout end
applylayout(::Type{typeof(vcat)}, _, ::AbstractBandedLayout) = VcatAlmostBandedLayout()
applylayout(::Type{typeof(vcat)}, _, _, ::AbstractBandedLayout) = VcatAlmostBandedLayout()
applylayout(::Type{typeof(vcat)}, _, _, _, ::AbstractBandedLayout) = VcatAlmostBandedLayout()
sublayout(::VcatAlmostBandedLayout, ::Type{<:Tuple{AbstractUnitRange{Int},AbstractUnitRange{Int}}}) = 
    VcatAlmostBandedLayout()

__two_arguments(z, y) = (y,z)
__two_arguments(z, y, x...) = (Vcat(reverse(x)..., y), z)
_two_arguments(A) = __two_arguments(reverse(arguments(A))...)

arguments(::VcatAlmostBandedLayout, A) = arguments(ApplyLayout{typeof(vcat)}(), A)

function almostbandwidths(::VcatAlmostBandedLayout, A)
    a,b = _two_arguments(A)
    m̃ = size(a,1)
    l,u = bandwidths(b)
    (l+m̃,u-m̃)
end

almostbandedrank(::VcatAlmostBandedLayout, A) = size(first(_two_arguments(A)),1)


function _copyto!(::AlmostBandedLayout, ::AlmostBandedLayout, dest::AbstractMatrix{T}, A::AbstractMatrix) where T
    m,n = size(dest)
    (m,n) == size(A) || throw(DimensionMismatch())
    copyto!(bandpart(dest), bandpart(A))
    copyto!(fillpart(dest), fillpart(A))
    dest
end

function _copyto!(::AlmostBandedLayout, ::VcatAlmostBandedLayout, dest::AbstractMatrix{T}, A::AbstractMatrix) where T
    m,n = size(dest)
    (m,n) == size(A) || throw(DimensionMismatch())
    a,b = _two_arguments(A)
    r = size(a,1)
    bands = bandpart(dest)
    U,V = arguments(fillpart(dest))
    U[1:r,1:r] = Eye{T}(r)
    zero!(view(U,r+1:m,:))
    copyto!(V, a)
    
    copyto!(@views(bands[r+1:end,:]), b)
    
    for j = 1:min(r+bandwidth(bands,2),size(bands,2))
        kr = colsupport(bands,j) ∩ (1:r)
        if !isempty(kr)
            bands[kr,j] .= view(a,kr,j)
        end
    end

    dest
end

function resize(A::LowRankMatrix, m::Integer, n::Integer)
    U,V = arguments(A)
    LowRankMatrix([U; Zeros(m-size(U,1),size(U,2))], [V Zeros(size(V,1), n-size(V,2))])
end

function resize(A::AlmostBandedMatrix, m::Integer, n::Integer)
    B = bandpart(A)
    F = fillpart(A)
    AlmostBandedMatrix(resize(B, m,n), resize(F,m,n))
end
    
function _cache(::VcatAlmostBandedLayout, A::AbstractArray{T}) where T
    r = almostbandedrank(A)
    l,u = almostbandwidths(A)
    data = AlmostBandedMatrix{T}(undef, (r,r+u), almostbandwidths(A), r)
    CachedArray(data, A, (0,0))
end

function resizedata!(::AlmostBandedLayout, ::VcatAlmostBandedLayout, C::CachedMatrix{T}, m::Integer, n::Integer) where T
    @boundscheck checkbounds(Bool, C, m, n) || throw(ArgumentError("Cannot resize beyound size of matrix"))
    A = C.array
    a,b = _two_arguments(A)
    r = almostbandedrank(C.data)

    # increase size of array if necessary
    μ,ν = C.datasize
    m,n = max(μ,m), max(ν,n)
    l,u = almostbandwidths(A)

    if (μ,ν) ≠ (m,n)
        λ,ω = almostbandwidths(C.data)
        if m ≥ size(C.data,1) || n ≥ size(C.data,2)
            N = 2*max(n,m+u,r)
            C.data = resize(C.data, N+λ, N)
        end
        if μ < r
            kr,jr = μ+1:min(r,m),1:ν
            copyto!(view(C.data,kr,jr), view(A,kr,jr))
        end
        if ν < r+ω+1
            kr,jr = 1:min(r,m),ν+1:min(r+ω+1,n)
            copyto!(view(C.data,kr,jr), view(A,kr,jr))
        end

        B,F = bandpart(C.data),fillpart(C.data)
        U,V = arguments(F)
        jr = max(ν+1,r+ω+2):n
        copyto!(view(V,:,jr), view(a,:,jr))
        zero!(view(U,max(μ+1,r+1):m,:))
        C_band = CachedArray(@view(B[r+1:end,:]), b, (max(0,μ-r),ν))
        resizedata!(C_band, max(0,m-r),n)
        C.datasize = (m,n)
    end
    
    C
end