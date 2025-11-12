struct BandedPlusSemiseparableMatrix{T} <: LayoutMatrix{T}
    bands::BandedMatrix{T}
    lowerfill::LowRankMatrix{T} 
    upperfill::LowRankMatrix{T}
end

BandedPlusSemiseparableMatrix(B, (U,V), (W,S)) = BandedPlusSemiseparableMatrix(B,  LowRankMatrix(U, V'), LowRankMatrix(W, S'))

size(A::BandedPlusSemiseparableMatrix) = size(A.bands)
copy(A::BandedPlusSemiseparableMatrix) = A # not mutable

function getindex(A::BandedPlusSemiseparableMatrix, k::Integer, j::Integer)
    if j > k 
        A.upperfill[k,j] + A.bands[k,j]
    elseif k > j
        A.lowerfill[k,j] + A.bands[k,j]
    else
        A.bands[k,j]
    end
end


"""
Represents factors matrix for QR for banded+semiseparable. we have

    for j = 0, we have A₀=tril(UV',-1) + B + triu(WS',1);
    F = qrfactUnblocked!(A₀).factors # full QR factors;
    As j increases,  for BandedPlusSemiseparableQRPerturbedFactors A:
    A[1:j,:] == F[1:j,:];
    A[:,1:j] == F[:,1:j];
    A[k, k] == F[k, k] for k < j;
    A[j+1:end,j+1:end] == A₀[j+1:end,j+1:end] + U[j+1:end,:]*Q*S[j+1:end]' + U[j+1:end,:]*K*U[j+1:end]'*A₀[j+1:end,j+1:end]
                          + U[j+1:end,:]*[Eₛ 0] + [Xₛ;0]*S[j+1:end]' + [Yₛ;0]*U[j+1:end]'*A₀[j+1:end,j+1:end] + [Zₛ 0;0 0]
"""

struct BandedPlusSemiseparableQRPerturbedFactors{T} <: LayoutMatrix{T}
    n::Int # matrix size
    r::Int # lower rank
    p::Int # upper rank
    l::Int # lower bandwidth
    m::Int # upper bandwidth
    U::Matrix{T} # n × r
    V::Matrix{T} # n × r
    W::Matrix{T} # n × (p+r)
    S::Matrix{T} # n × (p+r)
    B::BandedMatrix{T} # lower bandwidth l and upper bandwidth l + m

    Q::Matrix{T} # r × p
    K::Matrix{T} # r × r
    Eₛ::Matrix{T} # r × min(l+m,n)
    Xₛ::Matrix{T} # min(l,n) × p
    Yₛ::Matrix{T} # min(l,n) × r
    Zₛ::Matrix{T} # min(l,n) × min(l+m,n)

    j::Base.RefValue{Int} # how many columns have been upper-triangulised
end

BandedPlusSemiseparableMatrix(A::BandedPlusSemiseparableQRPerturbedFactors) = BandedPlusSemiseparableMatrix(A.B, (A.U,A.V), (A.W,A.S))
BandedPlusSemiseparableQRPerturbedFactors(A::BandedPlusSemiseparableMatrix) = BandedPlusSemiseparableQRPerturbedFactors(copy(A.lowerfill.args[1]), copy(A.lowerfill.args[2]'), copy(A.upperfill.args[1]), copy(A.upperfill.args[2]'), copy(A.bands))

size(A::BandedPlusSemiseparableQRPerturbedFactors) = (A.n,A.n)

function BandedPlusSemiseparableQRPerturbedFactors(U,V,W,S,B)
    n = size(U,1)
    r = size(U,2)
    p = size(W,2)
    l, m = bandwidths(B)
    A = tril(U*V',-1) + B + triu(W*S',1)
    AᵀU = (fast_UᵀA(U, V, W, S, B, 1))'
    BandedPlusSemiseparableQRPerturbedFactors(n,r,p,l,m,U,V,[W zeros(n,r)],[S AᵀU],BandedMatrix(B,(l,l+m)),
        zeros(r,p),zeros(r,r),zeros(r,min(l+m,n)),zeros(min(l,n),p),zeros(min(l,n),r),zeros(min(l,n),min(l+m,n)),Ref(0))
end

function getindex(A::BandedPlusSemiseparableQRPerturbedFactors, k::Integer, i::Integer)
    j = A.j[]
    if k > j && i > j
        AᵀU = (fast_UᵀA(A.U, A.V, A.W, A.S, A.B, j+1))'
        UQ = A.U[k,:]' * A.Q
        UK = A.U[k,:]' * A.K

        l = A.l[]
        m = A.m[]
        if k > i
            if k <= j + l
                A.U[k,:]' * A.V[i,:] + A.B[k,i] + UQ * A.S[i,1:A.p] + UK * AᵀU[i-j,:] + A.U[k,:]' * A.Eₛ[:,i-j] + A.Xₛ[k-j,:]' * A.S[i,1:A.p] + A.Yₛ[k-j,:]' * AᵀU[i-j,:] + A.Zₛ[k-j,i-j]
            else
                if i <= j + l + m
                    A.U[k,:]' * A.V[i,:] + A.B[k,i] + UQ * A.S[i,1:A.p] + UK * AᵀU[i-j,:] + A.U[k,:]' * A.Eₛ[:,i-j]
                else
                    A.U[k,:]' * A.V[i,:] + UQ * A.S[i,1:A.p] + UK * AᵀU[i-j,:] + A.B[k,i]
                end
            end
        elseif k < i
            if k <= j + l
                if i <= j + l + m
                    A.W[k,:]' * A.S[i,:] + A.B[k,i] + UQ * A.S[i,1:A.p] + UK * AᵀU[i-j,:] + A.U[k,:]' * A.Eₛ[:,i-j] + A.Xₛ[k-j,:]' * A.S[i,1:A.p] + A.Yₛ[k-j,:]' * AᵀU[i-j,:] + A.Zₛ[k-j,i-j]
                else
                    A.W[k,:]' * A.S[i,:] + A.B[k,i] + UQ * A.S[i,1:A.p] + UK * AᵀU[i-j,:] + A.Xₛ[k-j,:]' * A.S[i,1:A.p] + A.Yₛ[k-j,:]' * AᵀU[i-j,:]
                end
            else
                if i <= j + l + m
                    A.W[k,:]' * A.S[i,:] + A.B[k,i] + UQ * A.S[i,1:A.p] + UK * AᵀU[i-j,:] + A.U[k,:]' * A.Eₛ[:,i-j]
                else
                    A.W[k,:]' * A.S[i,:] + A.B[k,i] + UQ * A.S[i,1:A.p] + UK * AᵀU[i-j,:]
                end
            end
        else
            if k <= j + l
                A.B[k,i] + UQ * A.S[i,1:A.p] + UK * AᵀU[i-j,:] + A.U[k,:]' * A.Eₛ[:,i-j] + A.Xₛ[k-j,:]' * A.S[i,1:A.p] + A.Yₛ[k-j,:]' * AᵀU[i-j,:] + A.Zₛ[k-j,i-j]
            elseif i <= j + l + m
                A.B[k,i] + UQ * A.S[i,1:A.p] + UK * AᵀU[i-j,:] + A.U[k,:]' * A.Eₛ[:,i-j]
            else
                A.B[k,i] + UQ * A.S[i,1:A.p] + UK * AᵀU[i-j,:] 
            end
        end
    else
        if k > i
            A.U[k,:]' * A.V[i,:] + A.B[k,i]
        elseif k < i
            A.W[k,:]' * A.S[i,:] + A.B[k,i]
        else
            A.B[k,i]
        end
    end
end


function qr!(A::BandedPlusSemiseparableQRPerturbedFactors)
    n = A.n
    τ = zeros(n)
    if A.j[] != 0
        throw(ErrorException("Matrix has already been partially upper-triangularized"))
    end

    UᵀU = UᵀU_lookup_table(A)
    ūw̄_sum = ūw̄_sum_lookup_table(A)
    d_extra = d_extra_lookup_table(A)

    for i in 1 : n-1
        onestep_qr!(A, τ, UᵀU, ūw̄_sum, d_extra)
    end

    A.B[n,n] = A[n,n]
    A.j[] += 1

    QR(A, τ)
end

function qr(A::BandedPlusSemiseparableMatrix)
    F = qr!(BandedPlusSemiseparableQRPerturbedFactors(A))
    QR(BandedPlusSemiseparableMatrix(F.factors), F.τ)
end

function onestep_qr!(A, τ, UᵀU, ūw̄_sum, d_extra)
    k̄, b, p_new, τ_current= compute_Householder_vector(A, UᵀU) # I- τyy' where y = eⱼ₊₁ +  U⁽ʲ⁺²⁾k̄ + b and p_new will be the diagonal element on the current column 
    τ[A.j[]+1] = τ_current
    w̄₁, ū₁, d₁, f, d̄ = compute_d₁_and_d̄(A, b)
    c₁, c₂, c₃, c₄, c₅, c₆ = compute_variables_c(A, k̄, b, UᵀU)

    Q_prev = A.Q[:,:]
    K_prev = A.K[:,:]
    Eₛ_prev = A.Eₛ[:,:]
    Xₛ_prev = A.Xₛ[:,:]
    Yₛ_prev = A.Yₛ[:,:]
    Zₛ_prev = A.Zₛ[:,:]
    
    update_next_submatrix!(A, k̄, b, τ_current, w̄₁, ū₁, d₁, f, d̄, c₁, c₂, c₃, c₄, c₅, c₆)
    update_upper_triangular_part!(A, k̄, b, τ_current, w̄₁, ū₁, d₁, f, d̄, c₁, c₂, c₃, c₄, c₅, c₆, Q_prev, K_prev, Eₛ_prev, Xₛ_prev, Yₛ_prev, Zₛ_prev, ūw̄_sum, d_extra)
    update_lower_triangular_part!(A, p_new, k̄, b)

    A.j[] += 1

end

# the following are auxiliary functions:

function compute_Householder_vector(A, UᵀU)
    # compute Householder transformation I- τyy' where y = eⱼ₊₁ +  U⁽ʲ⁺²⁾k̄ + b

    # first express A[j+1:end,j+1] as p*eⱼ₊₁ + U⁽ʲ⁺²⁾k̄ + b
    j = A.j[]
    UᵀA_1 = (A.U[j+1:min(A.l+j+1,A.n),:])'*A.B[j+1:min(A.l+j+1,A.n),j+1] + UᵀU[j+2,:,:]*A.V[j+1,:] #the j+1th column of UᵀA

    k̄ = A.V[j+1,:] + A.Q * A.S[j+1,1:A.p] + A.K * UᵀA_1
    if A.l > 0 || A.m > 0
        k̄ += A.Eₛ[:,1]
    end
    b = A.B[j+2:min(A.l+j+1,A.n), j+1]
    if A.l > 0 || A.m > 0
        b[1:min(A.l-1,length(b))] += (A.Xₛ * A.S[j+1,1:A.p] + A.Yₛ * UᵀA_1 + A.Zₛ[:,1])[2:min(A.l,length(b)+1)]
    end
    p = A.B[j+1,j+1] + A.U[j+1,:]'A.Q * A.S[j+1,1:A.p] + A.U[j+1,:]'A.K*UᵀA_1
    if A.l > 0
        p += A.U[j+1,:]'A.Eₛ[:,1] + A.Xₛ[1,:]'A.S[j+1,1:A.p] + A.Yₛ[1,:]'UᵀA_1 + A.Zₛ[1,1]
    elseif A.m > 0
        p += A.U[j+1,:]'A.Eₛ[:,1]
    end

    # compute the length square of A[j+1:end,j+1]
    len_square = p^2 + k̄' * UᵀU[j+2,:,:] * k̄
    if A.l > 0
        len_square += k̄' * A.U[j+2:j+1+length(b),:]' * b + b' * A.U[j+2:j+1+length(b),:] * k̄ + b'b 
    end

    p_new = -sign(p) * sqrt(len_square) # the element on the diagonal after HT
    k̄ = k̄ / (p - p_new)
    b = b / (p - p_new)
    τ_current = 2 * (p - p_new)^2 / ((p-p_new)^2 + (len_square - p^2)) # the value τ for the current HT
    k̄, b, p_new, τ_current
end

function compute_d₁_and_d̄(A, b)
    j = A.j[]
    w̄₁ = A.W[j+1,1:A.p]
    ū₁ = A.U[j+1,:]
    d₁ = A.B[j+1,j+1:min(j+1+A.m, A.n)]
    d₁[1] = d₁[1] - w̄₁'*(A.S[j+1,1:A.p])
    f = (A.W[j+2:j+1+length(b),1:A.p])' * b
    index1 = j+2:j+1+length(b)
    index2 = j+1:min(j+1+A.m+A.l,A.n)
    tril_sub = [ (ii - jj) >= 1 for ii in index1, jj in index2 ]
    triu_sub = [ (jj - ii) >= 1 for ii in index1, jj in index2 ]
    A₀ = A.U[index1,:]*A.V[index2,:]'.*tril_sub + A.B[index1,index2] + A.W[index1,1:A.p]*A.S[index2,1:A.p]'.*triu_sub
    d̄ = (b'A₀ - f'*(A.S[index2,1:A.p])')'
    w̄₁, ū₁, d₁, f, d̄
end

function compute_variables_c(A, k̄, b, UᵀU)
    j = A.j[]
    UᵀU⁽²⁾ = UᵀU[j+2,:,:]

    c₁ = (A.Q)' * A.U[j+1,:] + (A.Q)' * UᵀU⁽²⁾ * k̄ + (A.Q)' * (A.U[j+2:j+1+length(b),:])' * b
    c₂ = (A.K)' * A.U[j+1,:] + (A.K)' * UᵀU⁽²⁾ * k̄ + (A.K)' * (A.U[j+2:j+1+length(b),:])' * b
    c₃ = A.U[j+1,:] + UᵀU⁽²⁾ * k̄ + (A.U[j+2:j+1+length(b),:])' * b

    Xₛ_valid = A.Xₛ[2:min(A.l, A.n-j),:]
    Yₛ_valid = A.Yₛ[2:min(A.l, A.n-j),:]
    Zₛ_valid = A.Zₛ[2:min(A.l, A.n-j),1:min(A.l+A.m, A.n-j)]
    c₄ = (Xₛ_valid)' * A.U[j+2:j+1+size(Xₛ_valid,1),:] * k̄ + (Xₛ_valid)' * b[1:size(Xₛ_valid,1)]
    c₅ = (Yₛ_valid)' * A.U[j+2:j+1+size(Yₛ_valid,1),:] * k̄ + (Yₛ_valid)' * b[1:size(Yₛ_valid,1)]
    c₆ = (Zₛ_valid)' * A.U[j+2:j+1+size(Zₛ_valid,1),:] * k̄ + (Zₛ_valid)' * b[1:size(Zₛ_valid,1)]
    if A.l > 0
        c₄ += A.Xₛ[1,:]
        c₅ += A.Yₛ[1,:]
        c₆ += A.Zₛ[1,1:min(A.l+A.m, A.n-j)]
    end

    c₁, c₂, c₃, c₄, c₅, c₆
end

function update_next_submatrix!(A, k̄, b, τ, w̄₁, ū₁, d₁, f, d̄, c₁, c₂, c₃, c₄, c₅, c₆)
    Q_prev = A.Q[:,:]
    K_prev = A.K[:,:]
    Eₛ_prev = A.Eₛ[:,:]
    Xₛ_prev = A.Xₛ[:,:]
    Yₛ_prev = A.Yₛ[:,:]
    Zₛ_prev = A.Zₛ[:,:]
  
    A.Q[:,:] = -τ*k̄*w̄₁' - τ*k̄*f' + Q_prev - τ*k̄*c₁' + K_prev*ū₁*w̄₁'-
                τ*k̄*c₂'*ū₁*w̄₁' - τ*k̄*c₄' - τ*k̄*c₅'*ū₁*w̄₁'
    A.K[:,:] = -τ*k̄*k̄' + K_prev - τ*k̄*c₂' - τ*k̄*c₅'

    A.Eₛ[:,1:length(d̄)-1] = -τ*k̄*(d̄[2:end])'
    A.Eₛ[:,1:length(d₁)-1] += (-τ*k̄ + K_prev*ū₁ - τ*k̄*c₂'*ū₁ - τ*k̄*c₅'*ū₁)*(d₁[2:end])'
    A.Eₛ[:,1:end-1] += Eₛ_prev[:,2:end] - τ*k̄*c₃'*Eₛ_prev[:,2:end]
    A.Eₛ[:,1:length(c₆)-1] += -τ*k̄*(c₆[2:end])'
    A.Eₛ[:,length(d̄):end] .= 0

    A.Xₛ[1:length(b),:] = b*(-τ*w̄₁' - τ*f' - τ*c₁' - τ*c₂'*ū₁*w̄₁' - τ*c₄' - τ*c₅'*ū₁*w̄₁')
    A.Xₛ[1:end-1,:] += Xₛ_prev[2:end,:] + Yₛ_prev[2:end,:]*ū₁*w̄₁'
    A.Xₛ[length(b)+1:end,:] .= 0

    A.Yₛ[1:length(b),:] = b*(-τ*k̄' - τ*c₂' - τ*c₅')
    A.Yₛ[1:end-1,:] += Yₛ_prev[2:end,:]
    A.Yₛ[length(b)+1:end,:] .= 0

    A.Zₛ[1:length(b), 1:length(d̄)-1] = -τ*b*(d̄[2:end])'
    A.Zₛ[1:length(b), 1:length(d₁)-1] += b*(-τ - τ*c₂'*ū₁ - τ*c₅'*ū₁)*(d₁[2:end])'
    A.Zₛ[1:length(b), 1:end-1] += -τ*b*c₃'*Eₛ_prev[:,2:end]
    A.Zₛ[1:end-1, 1:length(d₁)-1] += Yₛ_prev[2:end,:]*ū₁*(d₁[2:end])'
    A.Zₛ[1:end-1, 1:end-1] += Zₛ_prev[2:end, 2:end]
    A.Zₛ[1:length(b), 1:length(c₆)-1] += -τ*b*(c₆[2:end])'
    A.Zₛ[length(b)+1:end,:] .= 0
    A.Zₛ[:,length(d̄):end] .= 0

end

function update_upper_triangular_part!(A, k̄, b, τ, w̄₁, ū₁, d₁, f, d̄, c₁, c₂, c₃, c₄, c₅, c₆, Q, K, Eₛ, Xₛ, Yₛ, Zₛ, ūw̄_sum, d_extra)
    j = A.j[]
    
    β = (-τ*k̄' + ū₁'*K - τ*c₂' - τ*c₅')'
    if size(Yₛ, 1) > 0
        β += Yₛ[1,:]
    end

    α = (w̄₁' - τ*w̄₁' - τ*f' + ū₁'*Q - τ*c₁' - τ*c₄' + τ*k̄'*ū₁*w̄₁')'
    if size(Xₛ, 1) > 0
        α += Xₛ[1,:]
    end
    α += (-β'*ūw̄_sum[j+1,:,:])'

    d = -τ*d̄[2:end]
    d[1:length(d₁)-1] += (1 - τ + τ*k̄'*ū₁)*d₁[2:end]
    d[1:min(length(d),size(Eₛ,2)-1)] += ((ū₁' - τ*c₃')*Eₛ[:,2:min(length(d)+1,size(Eₛ,2))])'
    d[1:length(c₆)-1] += -τ*c₆[2:end]
    if size(Zₛ, 1) > 0
        d[1:min(length(d),size(Zₛ,2)-1)] += Zₛ[1, 2:min(length(d)+1,size(Zₛ,2))]
    end
    d_extra_current = d_extra[j+1]
    d[1:size(d_extra_current,2)] += (-β'*d_extra_current)'

    j = A.j[]
    A.W[j+1, 1:A.p] = α
    A.W[j+1, A.p+1:end] = β
    A.B[j+1, j+2:j+1+length(d)] = d
end

function update_lower_triangular_part!(A, p, k̄, b)
    j = A.j[]
    A.B[j+1, j+1] = p
    A.B[j+2:j+1+length(b), j+1] = b
    A.V[j+1, :] = k̄
end

function UᵀU_lookup_table(A)
    UᵀU = zeros(A.n, A.r, A.r)
    UᵀU_current = zeros(A.r, A.r)
    for i in A.n:-1:1
        UᵀU_current += A.U[i,:] * (A.U[i,:])'
        UᵀU[i,:,:] = UᵀU_current[:,:]
    end
    UᵀU
end

function ūw̄_sum_lookup_table(A)
    ūw̄_sum = zeros(A.n, A.r, A.p)
    ūw̄_sum_current = zeros(A.r, A.p)
    for t in 1:A.n
        ūw̄_sum[t,:,:] = ūw̄_sum_current[:,:]
        ūw̄_sum_current[:,:] += A.U[t,:] * (A.W[t,1:A.p])'
    end
    ūw̄_sum
end

function d_extra_lookup_table(A)
    d_extra = Vector{Matrix{eltype(A)}}()
    for i in 1:A.n
        d_extra_current = zeros(A.r, min(A.m, A.n-i))
        for t in max(1,i+1-A.m) : i-1
            d_extra_current += A.U[t,:] * (A.B[t, i+1:i+size(d_extra_current,2)])'
        end
        push!(d_extra, d_extra_current)
    end

    d_extra
end

function fast_UᵀA(U, V, W, S, B, j)
    # compute U[j,end]ᵀ*A[j:end,j:end] where A = tril(UV',-1) + B + triu(WS',1) in O(n)
    n = size(U,1)
    r = size(U,2)
    p = size(W,2)
    l, m = bandwidths(B)
    UᵀA = zeros(r, n+1-j)
    UᵀU = (U[j:end,:])'*U[j:end,:]
    UᵀW = zeros(r,p)
    for i in j:n
        UᵀU -= U[i,:] * (U[i,:])'
        UᵀA[:,i+1-j] = UᵀU*V[i,:] + (U[max(j,i-m) : min(i+l,n),:])'*B[max(j,i-m) : min(i+l,n),i] + UᵀW*S[i,:]
        UᵀW += U[i,:] * (W[i,:])'
    end
    UᵀA
end