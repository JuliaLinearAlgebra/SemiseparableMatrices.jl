"""
Represents factors matrix for QR for banded+semiseparable. we have

    for j = 0, we have A₀=tril(UV',-1) + B + triu(WS',1);
    F = qrfactUnblocked!(A₀).factors # full QR factors;
    As j increases,  for BandedPlusSemiseparableQRPerturbedFactors A:
    A[1:j,:] == F[1:j,:];
    A[:,1:j] == F[:,1:j];
    A[k, k] == F[k, k] for k < j;
    A[j+1:end,j+1:end] == A₀[j+1:end,j+1:end] + U[j+1:end,:]*Q*S[j+1:end,1:p]' + U[j+1:end,:]*K*U[j+1:end,:]'*A₀[j+1:end,j+1:end]
                          + U[j+1:end,:]*[Eₛ 0] + [Xₛ;0]*S[j+1:end,1:p]' + [Yₛ;0]*U[j+1:end]'*A₀[j+1:end,j+1:end] + [Zₛ 0;0 0]
"""

struct BandedPlusSemiseparableQRPerturbedFactors{T} <: LayoutMatrix{T}
    B::BandedMatrix{T, Matrix{T}, Base.OneTo{Int}} # lower bandwidth l and upper bandwidth l + m
    U::Matrix{T} # n × r
    V::Matrix{T} # n × r
    W::Matrix{T} # n × (p+r)
    S::Matrix{T} # n × (p+r)

    Q::Matrix{T} # r × p
    K::Matrix{T} # r × r
    Eₛ::Matrix{T} # r × min(l+m,n)
    Xₛ::Matrix{T} # min(l,n) × p
    Yₛ::Matrix{T} # min(l,n) × r
    Zₛ::Matrix{T} # min(l,n) × min(l+m,n)

    j::Base.RefValue{Int} # how many columns have been upper-triangulised
end

BandedPlusSemiseparableMatrix(A::BandedPlusSemiseparableQRPerturbedFactors) = BandedPlusSemiseparableMatrix(A.B, (A.U, A.V), (A.W, A.S))
BandedPlusSemiseparableQRPerturbedFactors(A::BandedPlusSemiseparableMatrix) = BandedPlusSemiseparableQRPerturbedFactors(copy(A.B), (copy(A.U), copy(A.V)), (copy(A.W), copy(A.S)))

size(A::BandedPlusSemiseparableQRPerturbedFactors) = size(A.B)

function BandedPlusSemiseparableQRPerturbedFactors(B, (U,V), (W,S))
    if size(U,1) == size(V,1) == size(W,1) == size(S,1) == size(B,1) == size(B,2) && size(U,2) == size(V,2) && size(W,2) == size(S,2)
        n, r = size(U)
        p = size(W,2)
        l, m = bandwidths(B)
        # A is tril(U*V',-1) + B + triu(W*S',1)
        AᵀU = (fast_UᵀA(U, V, W, S, B, 1))'
        BandedPlusSemiseparableQRPerturbedFactors(BandedMatrix(B,(l,l+m)),U,V,[W zeros(n,r)],[S AᵀU],
            zeros(r,p),zeros(r,r),zeros(r,min(l+m,n)),zeros(min(l,n),p),zeros(min(l,n),r),zeros(min(l,n),min(l+m,n)),Ref(0))
    else
        throw(DimensionMismatch("Dimensions are not compatible."))
    end
end

function getindex(A::BandedPlusSemiseparableQRPerturbedFactors, k::Integer, i::Integer)
    j = A.j[]
    p = size(A.W, 2) - size(A.U, 2) # W already been padded to have size n × (p+r)
    if k > j && i > j
        AᵀU = (fast_UᵀA(A.U, A.V, A.W, A.S, A.B, j+1))'
        UQ = view(A.U, k, :)' * A.Q
        UK = view(A.U, k, :)' * A.K

        l, m = bandwidths(A.B)
        m = m - l # B already been padded to have upper bandwidth l+m
        if k > i
            if k <= j + l
                view(A.U, k, :)' * view(A.V, i, :) + A.B[k,i] + UQ * view(A.S, i, 1:p) + UK * view(AᵀU, i-j, :) + view(A.U, k, :)' * view(A.Eₛ, :, i-j) + view(A.Xₛ, k-j, :)' * view(A.S, i, 1:p) + view(A.Yₛ, k-j, :)' * view(AᵀU, i-j, :) + A.Zₛ[k-j,i-j]
            else
                if i <= j + l + m
                    view(A.U, k, :)' * view(A.V, i, :) + A.B[k,i] + UQ * view(A.S, i, 1:p) + UK * view(AᵀU, i-j, :) + view(A.U, k, :)' * view(A.Eₛ, :, i-j)
                else
                    view(A.U, k, :)' * view(A.V, i, :) + UQ * view(A.S, i, 1:p) + UK * view(AᵀU, i-j, :) + A.B[k,i]
                end
            end
        elseif k < i
            if k <= j + l
                if i <= j + l + m
                    view(A.W, k, :)' * view(A.S, i, :) + A.B[k,i] + UQ * view(A.S, i, 1:p) + UK * view(AᵀU, i-j, :) + view(A.U, k, :)' * view(A.Eₛ, :, i-j) + view(A.Xₛ, k-j, :)' * view(A.S, i, 1:p) + view(A.Yₛ, k-j, :)' * view(AᵀU, i-j, :) + A.Zₛ[k-j,i-j]
                else
                    view(A.W, k, :)' * view(A.S, i, :) + A.B[k,i] + UQ * view(A.S, i, 1:p) + UK * view(AᵀU, i-j, :) + view(A.Xₛ, k-j, :)' * view(A.S, i, 1:p) + view(A.Yₛ, k-j, :)' * view(AᵀU, i-j, :)
                end
            else
                if i <= j + l + m
                    view(A.W, k, :)' * view(A.S, i, :) + A.B[k,i] + UQ * view(A.S, i, 1:p) + UK * view(AᵀU, i-j, :) + view(A.U, k, :)' * view(A.Eₛ, :, i-j)
                else
                    view(A.W, k, :)' * view(A.S, i, :) + A.B[k,i] + UQ * view(A.S, i, 1:p) + UK * view(AᵀU, i-j, :)
                end
            end
        else
            if k <= j + l
                A.B[k,i] + UQ * view(A.S, i, 1:p) + UK * view(AᵀU, i-j, :) + view(A.U, k, :)' * view(A.Eₛ, :, i-j) + view(A.Xₛ, k-j, :)' * view(A.S, i, 1:p) + view(A.Yₛ, k-j, :)' * view(AᵀU, i-j, :) + A.Zₛ[k-j,i-j]
            elseif i <= j + l + m
                A.B[k,i] + UQ * view(A.S, i, 1:p) + UK * view(AᵀU, i-j, :) + view(A.U, k, :)' * view(A.Eₛ, :, i-j)
            else
                A.B[k,i] + UQ * view(A.S, i, 1:p) + UK * view(AᵀU, i-j, :)
            end
        end
    else
        if k > i
            view(A.U, k, :)' * view(A.V, i, :) + A.B[k,i]
        elseif k < i
            view(A.W, k, :)' * view(A.S, i, :) + A.B[k,i]
        else
            A.B[k,i]
        end
    end
end

function qr!(A::BandedPlusSemiseparableQRPerturbedFactors{T}) where T
    if A.j[] != 0
        throw(ErrorException("Matrix has already been partially upper-triangularized"))
    end
    Q_prev = similar(A.Q)
    K_prev = similar(A.K)
    Eₛ_prev = similar(A.Eₛ)
    Xₛ_prev = similar(A.Xₛ)
    Yₛ_prev = similar(A.Yₛ)
    Zₛ_prev = similar(A.Zₛ)
    bandedplussemi_qr!(A,  zeros(T,size(A, 1)), UᵀU_lookup_table(A), ūw̄_sum_lookup_table(A), d_extra_lookup_table(A), Q_prev, K_prev, Eₛ_prev, Xₛ_prev, Yₛ_prev, Zₛ_prev)
end

function bandedplussemi_qr!(A, τ, tables...)
    n = size(A, 1)
    for i in 1 : n-1
        onestep_qr!(A, τ, tables...)
    end

    A.B[n,n] = A[n,n] # after n-1 HT the lower right 1 by 1 matrix is still represented by Q, K, etc. Simply get this value and assign it to the diagonal so the whole matrix is BPS now.
    A.j[] += 1
    QR(A, τ)
end

function qr(A::BandedPlusSemiseparableMatrix)
    F = qr!(BandedPlusSemiseparableQRPerturbedFactors(A))
    QR(BandedPlusSemiseparableMatrix(F.factors), F.τ)
end

function onestep_qr!(A, τ, UᵀU, ūw̄_sum, d_extra, Q_prev, K_prev, Eₛ_prev, Xₛ_prev, Yₛ_prev, Zₛ_prev)
    k̄, b, pivot_new, τ_current = compute_Householder_vector(A, UᵀU) # I- τyy' where y = eⱼ₊₁ +  U⁽ʲ⁺²⁾k̄ + b and p_new will be the diagonal element on the current column
    τ[A.j[]+1] = τ_current
    w̄₁, ū₁, d₁, f, d̄ = compute_d₁_and_d̄(A, b)
    c₁, c₂, c₃, c₄, c₅, c₆ = compute_variables_c(A, k̄, b, UᵀU)

    Q_prev .= A.Q
    K_prev .= A.K
    Eₛ_prev .= A.Eₛ
    Xₛ_prev .= A.Xₛ
    Yₛ_prev .= A.Yₛ
    Zₛ_prev .= A.Zₛ

    update_next_submatrix!(A, k̄, b, τ_current, w̄₁, ū₁, d₁, f, d̄, c₁, c₂, c₃, c₄, c₅, c₆, Q_prev, K_prev, Eₛ_prev, Xₛ_prev, Yₛ_prev, Zₛ_prev)
    update_upper_triangular_part!(A, k̄, b, τ_current, w̄₁, ū₁, d₁, f, d̄, c₁, c₂, c₃, c₄, c₅, c₆, Q_prev, K_prev, Eₛ_prev, Xₛ_prev, Yₛ_prev, Zₛ_prev, ūw̄_sum, d_extra)
    update_lower_triangular_part!(A, pivot_new, k̄, b)

    A.j[] += 1

end

# computation of qr when the original matrix is symmetric BPS

"""
Data structure for computing RQ when A is symmetric BPS. 

    For j = 0, we have R₀ = triu(B) + triu(WS',1);  
    Call the original U as U₀ It requires that S[1:r,:] = U₀ (as a result of A being symmetric) and precompute U = RU₀.
    As j increases,  for SymmetricBPSPerturbedRQ R:
    QR[i,i+1:n] = QR[i+1:n,i] = R[i+1:n,i] == B[i+1:n,i] + U[i+1:n,:]*V[i,:]',
    and QR[i, i] == R[i, i] for i = 1,...,j.
    R[j+1:end,j+1:end] == R₀[j+1:end,j+1:end] + R₀[j+1:end,j+1:end]*U[j+1:end,:]*Ω*U[j+1:end,:]'
                          + [Φₛ;0]*U[j+1:end,:]' + R₀[j+1:end,j+1:end]*U[j+1:end]*[Ψₛ 0] + [Λₛ 0;0 0]
    
    The information for Household transformations (Q) is stored in U₀, V,  and tril(B) the same way as in BandedPlusSemiseparableQRPerturbedFactors.
"""

struct SymmetricBPSPerturbedRQ{T} <: LayoutMatrix{T}
    B::BandedMatrix{T, Matrix{T}, Base.OneTo{Int}} # lower bandwidth l and upper bandwidth l+m
    U::Matrix{T} # n × r
    V::Matrix{T} # n × r
    W::Matrix{T} # n × (p+r)
    S::Matrix{T} # n × (p+r)

    Ω::Matrix{T} # r × r
    Φₛ::Matrix{T} # min(l,n) × r
    Ψₛ::Matrix{T} # r × min(l,n)
    Λₛ::Matrix{T} # min(l,n) × min(l,n)

    j::Base.RefValue{Int} # how many columns have been determined 
end

BandedPlusSemiseparableMatrix(R::SymmetricBPSPerturbedRQ) = BandedPlusSemiseparableMatrix(R.B, (R.U, R.V), (R.W, R.S))
SymmetricBPSPerturbedRQ(R::BandedPlusSemiseparableMatrix) = SymmetricBPSPerturbedRQ(copy(R.B), (copy(R.U), copy(R.V)), (copy(R.W), copy(R.S)))

size(R::SymmetricBPSPerturbedRQ) = size(R.B)

function SymmetricBPSPerturbedRQ(B, (U,V), (W,S))
    if size(U,1) == size(V,1) == size(W,1) == size(S,1) == size(B,1) == size(B,2) && size(U,2) == size(V,2) && size(W,2) == size(S,2)
        n, r = size(U)
        p = size(W,2)
        l, m = bandwidths(B)
        
        if p <= r
            throw(DimensionMismatch("The second dimension of S should be larger than that of U."))
        end

        if U != view(S, :, 1:r)
            throw(ErrorException("The first r column of S should be equal to U"))
        end

        if m < l
            throw(ErrorException("The upper bandwidth of B should be no less than the lower bandwidth"))
        end
        U_new = fast_RU(U, W, S, B, 1)
        SymmetricBPSPerturbedRQ(B,U_new,V,W,S,zeros(r,r),zeros(min(l,n),r),zeros(r,min(l,n)),zeros(min(l,n),min(l,n)),Ref(0))
    else
        throw(DimensionMismatch("Dimensions are not compatible."))
    end
end

function getindex(R::SymmetricBPSPerturbedRQ, k::Integer, i::Integer)
    j = R.j[]
    r = size(R.U, 2)
    p = size(R.W, 2) - size(R.U, 2)
    l, m = bandwidths(R.B)
    m = m - l 

    if k > j && i > j
        RU = fast_RU(R.U, R.W, R.S, R.B, j+1)
        if k > i
            if k <= j + l
                #(RU[k-j,:])' * R.Ω * R.S[i,1:r] + (R.Φₛ[k-j,:])' * R.S[i,1:r] + (RU[k-j,:])' * R.Ψₛ[:,i-j] + R.Λₛ[k-j,i-j]
                view(RU, k-j, :)' * R.Ω * view(R.S, i, 1:r) + view(R.Φₛ, k-j, :)' * view(R.S, i, 1:r) + view(RU, k-j, :)' * view(R.Ψₛ, :, i-j) + R.Λₛ[k-j,i-j]
            else
                if i <= j + l
                    #(RU[k-j,:])' * R.Ω * R.S[i,1:r] + (RU[k-j,:])' * R.Ψₛ[:,i-j]
                    view(RU, k-j, :)' * R.Ω * view(R.S, i, 1:r) + view(RU, k-j, :)' * view(R.Ψₛ, :, i-j)
                else
                    #(RU[k-j,:])' * R.Ω * R.S[i,1:r]
                    view(RU, k-j, :)' * R.Ω * view(R.S, i, 1:r)
                end
            end
        elseif k < i
            if k <= j + l
                if i <= j + l
                    #R.B[k,i] + R.W[k,:]' * R.S[i,:] + (RU[k-j,:])' * R.Ω * R.S[i,1:r] + (R.Φₛ[k-j,:])' * R.S[i,1:r] + (RU[k-j,:])' * R.Ψₛ[:,i-j] + R.Λₛ[k-j,i-j]
                    R.B[k,i] + view(R.W, k, :)' * view(R.S, i, :) + view(RU, k-j, :)' * R.Ω * view(R.S, i, 1:r) + view(R.Φₛ, k-j, :)' * view(R.S, i, 1:r) + view(RU, k-j, :)' * view(R.Ψₛ, :, i-j) + R.Λₛ[k-j,i-j]
                else
                    #R.B[k,i] + R.W[k,:]' * R.S[i,:] + (RU[k-j,:])' * R.Ω * R.S[i,1:r] + (R.Φₛ[k-j,:])' * R.S[i,1:r]
                    R.B[k,i] + view(R.W, k, :)' * view(R.S, i, :) + view(RU, k-j, :)' * R.Ω * view(R.S, i, 1:r) + view(R.Φₛ, k-j, :)' * view(R.S, i, 1:r)
                end
            else
                #R.B[k,i] + R.W[k,:]' * R.S[i,:] + (RU[k-j,:])' * R.Ω * R.S[i,1:r]
                R.B[k,i] + view(R.W, k, :)' * view(R.S, i, :) + view(RU, k-j, :)' * R.Ω * view(R.S, i, 1:r)
            end
        else
            if k <= j + l
                #R.B[k,i] + (RU[k-j,:])' * R.Ω * R.S[i,1:r] + (R.Φₛ[k-j,:])' * R.S[i,1:r] + (RU[k-j,:])' * R.Ψₛ[:,i-j] + R.Λₛ[k-j,i-j]
                R.B[k,i] + view(RU, k-j, :)' * R.Ω * view(R.S, i, 1:r) + view(R.Φₛ, k-j, :)' * view(R.S, i, 1:r) + view(RU, k-j, :)' * view(R.Ψₛ, :, i-j) + R.Λₛ[k-j,i-j]
            else
                #R.B[k,i] + (RU[k-j,:])' * R.Ω * R.S[i,1:r]
                R.B[k,i] + view(RU, k-j, :)' * R.Ω * view(R.S, i, 1:r)
            end
        end
    else
        if k > i
            #R.U[k,:]' * R.V[i,:] + R.B[k,i]
            view(R.U, k, :)' * view(R.V, i, :) + R.B[k,i]
        elseif k < i
            #R.U[i,:]' * R.V[k,:] + R.B[i,k]
            view(R.U, i, :)' * view(R.V, k, :) + R.B[i,k]
        else
            R.B[k,i]
        end
    end
end

function rq_mul!(R::SymmetricBPSPerturbedRQ{T}, τ::Vector{T}) where T
    if R.j[] != 0
        throw(ErrorException("RQ has already been partially computed"))
    end
    
    n = size(R, 1)
    UᵀU = UᵀU_lookup_table_rq(R)
    r⁽¹⁾ᵀU⁽²⁾ = r⁽¹⁾ᵀU⁽²⁾_lookup_table(R)
    for i in 1 : n-1
        onestep_rq!(R, τ, UᵀU, r⁽¹⁾ᵀU⁽²⁾)
    end

    R.B[n,n] = R[n,n] 
    R.j[] += 1
end

# Here R should be the factor matrix after the QR decomposition to a symmetric BPS matrix.
function rq_mul(R::BandedPlusSemiseparableMatrix, τ::Vector{T}) where T
    RR = SymmetricBPSPerturbedRQ(R)
    rq_mul!(RR, τ)
    symmetrize_from_lower!(RR.B)
    BandedPlusSemiseparableMatrix(RR.B, (RR.U, RR.V), (RR.V, RR.U))
end


function onestep_rq!(R, τ, UᵀU, r⁽¹⁾ᵀU⁽²⁾_table)
    j = R.j[]
    n = size(R, 1)
    l, m = bandwidths(R.B)
    b = R.B[j+2:min(l+j+1, n), j+1]
    k̄ = R.V[j+1,:]
    τ_current = τ[R.j[]+1]

    γ = compute_γ(R, b)
    δ₁, δ₂, δ₃, δ₄ = compute_variables_δ(R, k̄, b, UᵀU)

    update_lower_triangular_part_rq!(R, τ_current, k̄, b, γ, δ₁, δ₂, δ₃, δ₄)
    update_diagonal_rq!(R, τ_current, k̄, b, γ, δ₁, δ₂, δ₃, δ₄, r⁽¹⁾ᵀU⁽²⁾_table)
    update_next_submatrix_rq!(R, τ_current, k̄, b, γ, δ₁, δ₂, δ₃, δ₄)
    R.j[] += 1
end

# the following are auxiliary functions:

function compute_Householder_vector(A, UᵀU)
    # compute Householder transformation I- τyy' where y = eⱼ₊₁ +  U⁽ʲ⁺²⁾k̄ + b

    # first express A[j+1:end,j+1] as p*eⱼ₊₁ + U⁽ʲ⁺²⁾k̄ + b
    j = A.j[]
    n = size(A, 1)
    p = size(A.W, 2) - size(A.U, 2) # W already been padded to have size n × (p+r)
    l, m = bandwidths(A.B)
    m = m - l # B already been padded to have upper bandwidth l+m
    T = eltype(A)
    # UᵀA_1 = (A.U[j+1:min(A.l+j+1,A.n),:])'*A.B[j+1:min(A.l+j+1,A.n),j+1] + UᵀU[j+2,:,:]*A.V[j+1,:], the j+1th column of UᵀA
    UᵀA_1 = (view(A.U, j+1:min(l+j+1, n),:))' * view(A.B, j+1:min(l+j+1, n) , j+1) # matrix allocation
    mul!(UᵀA_1, view(UᵀU, j+2 , : , :), view(A.V, j+1, :), one(T), one(T))

    #k̄ = A.V[j+1,:] + A.Q * A.S[j+1,1:A.p] + A.K * UᵀA_1
    k̄ = A.K * UᵀA_1 # matrix accolation
    mul!(k̄, A.Q, view(A.S, j+1, 1:p), one(T), one(T))
    k̄ .+= view(A.V, j+1, :)
    if l > 0 || m > 0
        k̄ .+= view(A.Eₛ, :, 1)
    end
    b = A.B[j+2:min(l+j+1, n), j+1] # matrix allocation
    if l > 0 || m > 0
        #b[1:min(A.l-1,length(b))] += (A.Xₛ * A.S[j+1,1:A.p] + A.Yₛ * UᵀA_1 + A.Zₛ[:,1])[2:min(A.l,length(b)+1)]
        kr = 2:min(l,length(b)+1)
        v = view(b, kr .- 1)
        mul!(v, view(A.Xₛ, kr,:),  view(A.S, j+1, 1:p), one(T), one(T))
        mul!(v, view(A.Yₛ,kr,:),  UᵀA_1, one(T), one(T))
        v .+= view(A.Zₛ, kr, 1)
    end
    #pivot = A.B[j+1,j+1] + A.U[j+1,:]'A.Q * A.S[j+1,1:A.p] + A.U[j+1,:]'A.K*UᵀA_1
    pivot = A.B[j+1,j+1] + (view(A.U, j+1, :))' * A.Q * view(A.S, j+1, 1:p) + view(A.U, j+1, :)' * A.K * UᵀA_1
    if l > 0
        pivot += (view(A.U, j+1 , :))' * view(A.Eₛ, :, 1) + (view(A.Xₛ, 1 , :))' * view(A.S, j+1 , 1:p) + (view(A.Yₛ, 1, :))' * UᵀA_1 + A.Zₛ[1,1]
    elseif m > 0
        pivot += (view(A.U, j+1, :))' * view(A.Eₛ, :, 1)
    end

    # compute the length square of A[j+1:end,j+1]
    len_square = pivot^2 + k̄' * view(UᵀU, j+2, :, :) * k̄
    if l > 0
        len_square += k̄' * (view(A.U, j+2:j+1+length(b), :))' * b + b' * view(A.U, j+2:j+1+length(b), :) * k̄ + b'b
    end

    pivot_new = -sign(pivot) * sqrt(len_square) # the element on the diagonal after HT
    k̄ ./= (pivot - pivot_new)
    b ./= (pivot - pivot_new)
    τ_current = 2 * (pivot - pivot_new)^2 / ((pivot-pivot_new)^2 + (len_square - pivot^2)) # the value τ for the current HT
    k̄, b, pivot_new, τ_current
end

function compute_d₁_and_d̄(A, b)
    j = A.j[]
    n = size(A, 1)
    p = size(A.W, 2) - size(A.U, 2) # W already been padded to have size n × (p+r)
    l, m = bandwidths(A.B)
    m = m - l # B already been padded to have upper bandwidth l+m
    T = eltype(A)
    w̄₁ = view(A.W, j+1, 1:p)
    ū₁ = view(A.U, j+1, :)
    d₁ = A.B[j+1,j+1:min(j+1+m, n)] # matrix allocation
    d₁[1] = d₁[1] - w̄₁'*(view(A.S, j+1, 1:p))
    f = (view(A.W, j+2:j+1+length(b), 1:p))' * b # matrix allocation
    index1 = j+2:j+1+length(b)
    index2 = j+1:min(j+1+m+l, n)
    tril_sub = [ (ii - jj) >= 1 for ii in index1, jj in index2 ]
    triu_sub = [ (jj - ii) >= 1 for ii in index1, jj in index2 ]
    #A₀ = A.U[index1,:]*A.V[index2,:]'.*tril_sub + A.B[index1,index2] + A.W[index1,1:A.p]*A.S[index2,1:A.p]'.*triu_sub
    A₀ = view(A.U, index1, :) * view(A.V, index2, :)' # matrix allocation
    A₀ .*= tril_sub
    A₀ .+= view(A.B, index1, index2)
    A₀_up = view(A.W, index1, 1:p) * view(A.S, index2, 1:p)' # matrix allocation
    A₀_up .*= triu_sub
    A₀ .+= A₀_up
    #d̄ = (b'A₀ - f'*(A.S[index2,1:A.p])')'
    d̄ = A₀'b # matrix allocation
    mul!(d̄, view(A.S, index2, 1:p), f, -one(T), one(T))
    w̄₁, ū₁, d₁, f, d̄
end

function compute_variables_c(A, k̄, b, UᵀU)
    j = A.j[]
    n = size(A, 1)
    l, m = bandwidths(A.B)
    m = m - l # B already been padded to have upper bandwidth l+m
    T = eltype(A)
    UᵀU⁽²⁾ = view(UᵀU, j+2, :, :)

    #c₁ = (A.Q)' * A.U[j+1,:] + (A.Q)' * UᵀU⁽²⁾ * k̄ + (A.Q)' * (A.U[j+2:j+1+length(b),:])' * b
    c₁ = (A.Q)' * view(A.U, j+1, :) # matrix allocation
    mul!(c₁, (A.Q)' * UᵀU⁽²⁾, k̄, one(T), one(T))
    mul!(c₁, (A.Q)' * (view(A.U, j+2:j+1+length(b), :))', b, one(T), one(T))
    #c₂ = (A.K)' * A.U[j+1,:] + (A.K)' * UᵀU⁽²⁾ * k̄ + (A.K)' * (A.U[j+2:j+1+length(b),:])' * b
    c₂ = (A.K)' * view(A.U, j+1, :) # matrix allocation
    mul!(c₂, (A.K)' * UᵀU⁽²⁾, k̄, one(T), one(T))
    mul!(c₂, (A.K)' * (view(A.U, j+2:j+1+length(b), :))', b, one(T), one(T))
    #c₃ = A.U[j+1,:] + UᵀU⁽²⁾ * k̄ + (A.U[j+2:j+1+length(b),:])' * b
    c₃ = UᵀU⁽²⁾ * k̄ # matrix allocation
    c₃ .+= view(A.U, j+1, :)
    mul!(c₃, view(A.U, j+2:j+1+length(b), :)', b, one(T), one(T))

    Xₛ_valid = view(A.Xₛ, 2:min(l, n-j), :)
    Yₛ_valid = view(A.Yₛ, 2:min(l, n-j), :)
    Zₛ_valid = view(A.Zₛ, 2:min(l, n-j), 1:min(l+m, n-j))
    #c₄ = (Xₛ_valid)' * A.U[j+2:j+1+size(Xₛ_valid,1),:] * k̄ + (Xₛ_valid)' * b[1:size(Xₛ_valid,1)]
    c₄ = (Xₛ_valid)' * view(b, 1:size(Xₛ_valid,1)) # matrix allocation
    mul!(c₄, (Xₛ_valid)' * view(A.U, j+2:j+1+size(Xₛ_valid,1), :), k̄, one(T), one(T))
    #c₅ = (Yₛ_valid)' * A.U[j+2:j+1+size(Yₛ_valid,1),:] * k̄ + (Yₛ_valid)' * b[1:size(Yₛ_valid,1)]
    c₅ = (Yₛ_valid)' * view(b, 1:size(Yₛ_valid,1)) # matrix allocation
    mul!(c₅, (Yₛ_valid)' * view(A.U, j+2:j+1+size(Yₛ_valid,1), :), k̄, one(T), one(T))
    #c₆ = (Zₛ_valid)' * A.U[j+2:j+1+size(Zₛ_valid,1),:] * k̄ + (Zₛ_valid)' * b[1:size(Zₛ_valid,1)]
    c₆ = (Zₛ_valid)' * view(b, 1:size(Zₛ_valid,1)) # matrix allocation
    mul!(c₆, (Zₛ_valid)' * view(A.U, j+2:j+1+size(Zₛ_valid,1), :), k̄, one(T), one(T))
    if l > 0
        #c₄ += A.Xₛ[1,:]
        c₄ .+= view(A.Xₛ, 1, :)
        #c₅ += A.Yₛ[1,:]
        c₅ .+= view(A.Yₛ, 1, :)
        #c₆ += A.Zₛ[1,1:min(A.l+A.m, A.n-j)]
        c₆ .+= view(A.Zₛ, 1, 1:min(l+m, n-j))
    end

    c₁, c₂, c₃, c₄, c₅, c₆
end

function update_next_submatrix!(A, k̄, b, τ, w̄₁, ū₁, d₁, f, d̄, c₁, c₂, c₃, c₄, c₅, c₆, Q_prev, K_prev, Eₛ_prev, Xₛ_prev, Yₛ_prev, Zₛ_prev)
    # A.Q .= -τ*k̄*w̄₁' - τ*k̄*f' + Q_prev - τ*k̄*c₁' + K_prev*ū₁*w̄₁'-
    #             τ*k̄*c₂'*ū₁*w̄₁' - τ*k̄*c₄' - τ*k̄*c₅'*ū₁*w̄₁'
    T = eltype(A)
    mul!(A.Q, k̄, w̄₁', -τ, one(T))
    mul!(A.Q, k̄, f', -τ, one(T))
    mul!(A.Q, k̄, c₁', -τ, one(T))
    mul!(A.Q, K_prev, ū₁*w̄₁', one(T), one(T)) # TODO: write to tem buffer?
    mul!(A.Q, k̄, w̄₁', -τ*(c₂'*ū₁+c₅'*ū₁), one(T))
    mul!(A.Q, k̄, c₄', -τ, one(T))

    # A.K .= -τ*k̄*k̄' + K_prev - τ*k̄*c₂' - τ*k̄*c₅'
    mul!(A.K, k̄, k̄', -τ, one(T))
    mul!(A.K, k̄, c₂', -τ, one(T))
    mul!(A.K, k̄, c₅', -τ, one(T))

    #A.Eₛ[:,1:length(d̄)-1] = -τ*k̄*(d̄[2:end])'
    mul!(view(A.Eₛ, :, 1:length(d̄)-1), k̄, view(d̄, 2:length(d̄))', -τ, zero(T))
    #A.Eₛ[:,1:length(d₁)-1] += (-τ*k̄ + K_prev*ū₁ - τ*k̄*c₂'*ū₁ - τ*k̄*c₅'*ū₁)*(d₁[2:end])'
    mul!(view(A.Eₛ, :, 1:length(d₁)-1), k̄, view(d₁, 2:length(d₁))', -τ, one(T))
    mul!(view(A.Eₛ, :, 1:length(d₁)-1), K_prev*ū₁, view(d₁, 2:length(d₁))', one(T), one(T))
    mul!(view(A.Eₛ, :, 1:length(d₁)-1), k̄*c₂'*ū₁, view(d₁, 2:length(d₁))',  -τ, one(T))
    mul!(view(A.Eₛ, :, 1:length(d₁)-1), k̄*c₅'*ū₁, view(d₁, 2:length(d₁))',  -τ, one(T))
    #A.Eₛ[:,1:end-1] += Eₛ_prev[:,2:end] - τ*k̄*c₃'*Eₛ_prev[:,2:end]
    view(A.Eₛ, :, 1:size(A.Eₛ,2)-1) .+= view(Eₛ_prev, :, 2:size(Eₛ_prev,2))
    mul!(view(A.Eₛ, :, 1:size(A.Eₛ,2)-1), k̄*c₃', view(Eₛ_prev, :, 2:size(Eₛ_prev,2)), -τ, one(T))
    #A.Eₛ[:,1:length(c₆)-1] += -τ*k̄*(c₆[2:end])'
    mul!(view(A.Eₛ, :, 1:length(c₆)-1), k̄, view(c₆, 2:length(c₆))', -τ, one(T))
    #A.Eₛ[:,length(d̄):end] .= 0
    view(A.Eₛ, :, length(d̄):size(A.Eₛ,2)) .= zero(eltype(A))

    #A.Xₛ[1:length(b),:] = b*(-τ*w̄₁' - τ*f' - τ*c₁' - τ*c₂'*ū₁*w̄₁' - τ*c₄' - τ*c₅'*ū₁*w̄₁')
    mul!(view(A.Xₛ, 1:length(b),:), b, w̄₁', -τ, zero(T))
    mul!(view(A.Xₛ, 1:length(b),:), b, f', -τ, one(T))
    mul!(view(A.Xₛ, 1:length(b),:), b, c₁', -τ, one(T))
    mul!(view(A.Xₛ, 1:length(b),:), b, c₂'*ū₁*w̄₁', -τ, one(T))
    mul!(view(A.Xₛ, 1:length(b),:), b, c₄', -τ, one(T))
    mul!(view(A.Xₛ, 1:length(b),:), b, c₅'*ū₁*w̄₁', -τ, one(T))
    #A.Xₛ[1:end-1,:] += Xₛ_prev[2:end,:] + Yₛ_prev[2:end,:]*ū₁*w̄₁'
    view(A.Xₛ, 1:size(A.Xₛ,1)-1, :) .+= view(Xₛ_prev, 2:size(Xₛ_prev,1), :)
    mul!(view(A.Xₛ, 1:size(A.Xₛ,1)-1, :), view(Yₛ_prev, 2:size(Yₛ_prev,1), :), ū₁*w̄₁', one(T), one(T))
    #A.Xₛ[length(b)+1:end,:] .= zero(eltype(A))
    view(A.Xₛ, length(b)+1:size(A.Xₛ,1), :) .= zero(eltype(A))

    #A.Yₛ[1:length(b),:] = b*(-τ*k̄' - τ*c₂' - τ*c₅')
    mul!(view(A.Yₛ, 1:length(b), :), b, k̄', -τ, zero(T))
    mul!(view(A.Yₛ, 1:length(b), :), b, c₂', -τ, one(T))
    mul!(view(A.Yₛ, 1:length(b), :), b, c₅', -τ, one(T))
    #A.Yₛ[1:end-1,:] += Yₛ_prev[2:end,:]
    view(A.Yₛ, 1:size(A.Yₛ,1)-1, :) .+= view(Yₛ_prev, 2:size(Yₛ_prev,1), :)
    #A.Yₛ[length(b)+1:end,:] .= 0
    view(A.Yₛ, length(b)+1:size(A.Yₛ,1), :) .= zero(eltype(A))

    #A.Zₛ[1:length(b), 1:length(d̄)-1] = -τ*b*(d̄[2:end])'
    mul!(view(A.Zₛ, 1:length(b), 1:length(d̄)-1), b, view(d̄, 2:length(d̄))', -τ, zero(T))
    #A.Zₛ[1:length(b), 1:length(d₁)-1] += b*(-τ - τ*c₂'*ū₁ - τ*c₅'*ū₁)*(d₁[2:end])'
    mul!(view(A.Zₛ, 1:length(b), 1:length(d₁)-1), b, view(d₁, 2:length(d₁))', -τ, one(T))
    mul!(view(A.Zₛ, 1:length(b), 1:length(d₁)-1), b*c₂'*ū₁, view(d₁, 2:length(d₁))', -τ, one(T))
    mul!(view(A.Zₛ, 1:length(b), 1:length(d₁)-1), b*c₅'*ū₁, view(d₁, 2:length(d₁))', -τ, one(T))
    #A.Zₛ[1:length(b), 1:end-1] += -τ*b*c₃'*Eₛ_prev[:,2:end]
    mul!(view(A.Zₛ, 1:length(b), 1:size(A.Zₛ,2)-1), b*c₃', view(Eₛ_prev, :, 2:size(Eₛ_prev, 2)), -τ, one(T))
    #A.Zₛ[1:end-1, 1:length(d₁)-1] += Yₛ_prev[2:end,:]*ū₁*(d₁[2:end])'
    mul!(view(A.Zₛ, 1:size(A.Zₛ,1)-1, 1:length(d₁)-1), view(Yₛ_prev, 2:size(Yₛ_prev,1), :)*ū₁, view(d₁, 2:length(d₁))', one(T), one(T))
    #A.Zₛ[1:end-1, 1:end-1] += Zₛ_prev[2:end, 2:end]
    view(A.Zₛ, 1:size(A.Zₛ,1)-1, 1:size(A.Zₛ,2)-1) .+= view(Zₛ_prev, 2:size(Zₛ_prev,1), 2:size(Zₛ_prev,2))
    #A.Zₛ[1:length(b), 1:length(c₆)-1] += -τ*b*(c₆[2:end])'
    mul!(view(A.Zₛ, 1:length(b), 1:length(c₆)-1), b, view(c₆, 2:length(c₆))', -τ, one(T))
    #A.Zₛ[length(b)+1:end,:] .= 0
    view(A.Zₛ, length(b)+1:size(A.Zₛ,1),:) .= zero(eltype(A))
    #A.Zₛ[:,length(d̄):end] .= 0
    view(A.Zₛ, :, length(d̄):size(A.Zₛ,2)) .= zero(eltype(A))

end

function update_upper_triangular_part!(A, k̄, b, τ, w̄₁, ū₁, d₁, f, d̄, c₁, c₂, c₃, c₄, c₅, c₆, Q, K, Eₛ, Xₛ, Yₛ, Zₛ, ūw̄_sum, d_extra)
    j = A.j[]
    p = size(A.W, 2) - size(A.U, 2) # W already been padded to have size n × (p+r)
    T = eltype(A)
    #β = (-τ*k̄' + ū₁'*K - τ*c₂' - τ*c₅')'
    β = -τ*k̄ # matrix allocation
    mul!(β, K', ū₁, one(T), one(T))
    mul!(β, I, c₂, -τ, one(T))
    mul!(β, I, c₅, -τ, one(T))
    if size(Yₛ, 1) > 0
        β .+= view(Yₛ, 1, :)
    end

    #α = (w̄₁' - τ*w̄₁' - τ*f' + ū₁'*Q - τ*c₁' - τ*c₄' + τ*k̄'*ū₁*w̄₁')'
    α = (1-τ)*w̄₁ # matrix allocation
    mul!(α, I, f, -τ, one(T))
    mul!(α, Q', ū₁, one(T), one(T))
    mul!(α, I, c₁, -τ, one(T))
    mul!(α, I, c₄, -τ, one(T))
    mul!(α, w̄₁*ū₁', k̄, τ, one(T))
    if size(Xₛ, 1) > 0
        α .+= view(Xₛ, 1, :)
    end
    #α += (-β'*ūw̄_sum[j+1,:,:])'
    mul!(α, view(ūw̄_sum, j+1, :, :)', β, -one(T), one(T))

    d = -τ * d̄[2:end] # matrix allocation
    #d[1:length(d₁)-1] += (1 - τ + τ*k̄'*ū₁)*d₁[2:end]
    mul!(view(d, 1:length(d₁)-1), I, view(d₁, 2:length(d₁)), 1 - τ + τ*k̄'*ū₁, one(T))
    #d[1:min(length(d),size(Eₛ,2)-1)] += ((ū₁' - τ*c₃')*Eₛ[:,2:min(length(d)+1,size(Eₛ,2))])'
    mul!(view(d, 1:min(length(d),size(Eₛ,2)-1)), view(Eₛ, :, 2:min(length(d)+1,size(Eₛ,2)))', ū₁-τ*c₃, one(T), one(T))
    #d[1:length(c₆)-1] += -τ*c₆[2:end]
    mul!(view(d, 1:length(c₆)-1), I, view(c₆, 2:length(c₆)), -τ, one(T))
    if size(Zₛ, 1) > 0
        #d[1:min(length(d),size(Zₛ,2)-1)] += Zₛ[1, 2:min(length(d)+1,size(Zₛ,2))]
        d[1:min(length(d),size(Zₛ,2)-1)] .+= view(Zₛ, 1, 2:min(length(d)+1,size(Zₛ,2)))
    end
    d_extra_current = d_extra[j+1]
    #d[1:size(d_extra_current,2)] += (-β'*d_extra_current)'
    mul!(view(d, 1:size(d_extra_current,2)), d_extra_current', β, -one(T), one(T))

    j = A.j[]
    #A.W[j+1, 1:p] = α
    view(A.W, j+1, 1:p) .= α
    #A.W[j+1, p+1:end] = β
    view(A.W, j+1, (p+1):size(A.W,2)) .= β
    #A.B[j+1, j+2:j+1+length(d)] = d
    view(A.B, j+1, j+2:j+1+length(d)) .= d
end

function update_lower_triangular_part!(A, pivot, k̄, b)
    j = A.j[]
    A.B[j+1, j+1] = pivot
    #A.B[j+2:j+1+length(b), j+1] = b
    view(A.B, j+2:j+1+length(b), j+1) .= b
    #A.V[j+1, :] = k̄
    view(A.V, j+1, :) .= k̄
end

function UᵀU_lookup_table(A)
    n, r = size(A.U)
    T = eltype(A)
    UᵀU = zeros(T, n, r, r)
    UᵀU_current = zeros(T, r, r)
    for i in n:-1:1
        #UᵀU_current += A.U[i,:] * (A.U[i,:])'
        mul!(UᵀU_current, view(A.U, i, :), (view(A.U, i, :))', one(T), one(T))
        #UᵀU[i,:,:] .= UᵀU_current
        view(UᵀU, i, :, :) .= UᵀU_current
    end
    UᵀU
end

function ūw̄_sum_lookup_table(A)
    n, r = size(A.U)
    p = size(A.W, 2) - size(A.U, 2) # W already been padded to have size n × (p+r)
    T = eltype(A)
    ūw̄_sum = zeros(T, n, r, p)
    ūw̄_sum_current = zeros(T, r, p)
    for t in 1:n
        #ūw̄_sum[t,:,:] .= ūw̄_sum_current
        view(ūw̄_sum, t, :, :) .= ūw̄_sum_current
        #ūw̄_sum_current[:,:] .+= A.U[t,:] * (A.W[t,1:A.p])'
        mul!(ūw̄_sum_current, view(A.U, t, :), (view(A.W, t, 1:p))', one(T), one(T))
    end
    ūw̄_sum
end

function d_extra_lookup_table(A)
    n, r = size(A.U)
    l, m = bandwidths(A.B)
    m = m - l # B already been padded to have upper bandwidth l+m
    T = eltype(A)
    d_extra = Vector{Matrix{T}}()
    for i in 1:n
        d_extra_current = zeros(eltype(A), r, min(m, n-i)) # matrix allocation
        for t in max(1,i+1-m) : i-1
            #d_extra_current .+= A.U[t,:] * (A.B[t, i+1:i+size(d_extra_current,2)])'
            mul!(d_extra_current, view(A.U, t, :), (view(A.B, t, i+1:i+size(d_extra_current,2)))', one(T), one(T))
        end
        push!(d_extra, d_extra_current)
    end

    d_extra
end

function fast_UᵀA(U, V, W, S, B, j)
    # compute U[j,end]ᵀ*A[j:end,j:end] where A = tril(UV',-1) + B + triu(WS',1) in O(n)
    n, r = size(U)
    p = size(W,2)
    l, m = bandwidths(B)
    T = eltype(U)
    UᵀA = zeros(T, r, n+1-j)
    UᵀU = (view(U, j:n, :))' * view(U, j:n, :)
    UᵀW = zeros(T, r, p)
    for i in j:n
        #UᵀU -= U[i,:] * (U[i,:])'
        mul!(UᵀU, view(U, i, :), (view(U, i, :))', -one(T), one(T))
        #UᵀA[:,i+1-j] = UᵀU*V[i,:] + (U[max(j,i-m) : min(i+l,n),:])'*B[max(j,i-m) : min(i+l,n),i] + UᵀW*S[i,:]
        mul!(view(UᵀA, :, i+1-j), UᵀU, view(V, i, :), one(T), zero(T))
        mul!(view(UᵀA, :, i+1-j), (view(U, max(j,i-m) : min(i+l,n), :))', view(B, max(j,i-m) : min(i+l,n), i), one(T), one(T))
        mul!(view(UᵀA, :, i+1-j), UᵀW, view(S, i, :), one(T), one(T))
        #UᵀW += U[i,:] * (W[i,:])'
        mul!(UᵀW, view(U, i, :), (view(W, i, :))', one(T), one(T))
    end
    UᵀA
end

function compute_γ(R, b)
    T = eltype(R)
    j = R.j[]
    γ = zeros(T, length(b)+1)
    for i in 1:length(b)
        #γ[1:i+1] += b[i]*R.B[1+j:1+i+j, 1+i+j]
        view(γ, 1:i+1) .+= b[i]*view(R.B, 1+j:1+i+j, 1+i+j)
        #γ[1:i] += b[i]*R.W[1+j:i+j,:]*R.S[1+i+j,:]
        mul!(view(γ, 1:i), view(R.W,1+j:i+j,:), view(R.S,1+i+j,:), b[i], one(T))
    end
    γ
end

function compute_variables_δ(R, k̄, b, UᵀU)
    j = R.j[]
    n, r = size(R.U)
    l,m = bandwidths(R.B)
    #UᵀU⁽²⁾ = UᵀU[j+2, :, :]
    UᵀU⁽²⁾ = view(UᵀU, j+2, :, :)

    #δ₂ = R.S[1+j,1:r] + UᵀU⁽²⁾*k̄ + (R.S[2+j:length(b)+1+j,1:r])'*b
    δ₂ = view(R.S, 1+j, 1:r) + UᵀU⁽²⁾*k̄ + (view(R.S, 2+j:length(b)+1+j, 1:r))'*b
    δ₁ = R.Ω * δ₂

    #Ψ_valid = R.Ψₛ[:,1:min(l,n-j)]
    Ψ_valid = view(R.Ψₛ, :, 1:min(l,n-j))
    #δ₃ = Ψ_valid[:,2:end]*R.S[2+j:size(Ψ_valid,2)+j,1:r]*k̄ + Ψ_valid[:,2:end]*b[1:size(Ψ_valid,2)-1]
    δ₃ = view(Ψ_valid, :, 2:size(Ψ_valid,2))*view(R.S, 2+j:size(Ψ_valid,2)+j, 1:r)*k̄ + view(Ψ_valid, :, 2:size(Ψ_valid,2))*view(b, 1:size(Ψ_valid,2)-1)
    if l > 0
        #δ₃ += Ψ_valid[:,1]
        δ₃ .+= view(Ψ_valid, :, 1)
    end

    #Λ_valid = R.Λₛ[1:min(l,n-j), 1:min(l,n-j)]
    Λ_valid = view(R.Λₛ, 1:min(l,n-j), 1:min(l,n-j))
    #δ₄ = Λ_valid[:,2:end]*R.S[2+j:size(Λ_valid,2)+j,1:r]*k̄ + Λ_valid[:,2:end]*b[1:size(Λ_valid,2)-1]
    δ₄ = view(Λ_valid, :, 2:size(Λ_valid,2))*view(R.S, 2+j:size(Λ_valid,2)+j, 1:r)*k̄ + view(Λ_valid, :, 2:size(Λ_valid,2))*view(b, 1:size(Λ_valid,2)-1)
    if l > 0
        #δ₄ += Λ_valid[:,1]
        δ₄ .+= view(Λ_valid, :, 1)
    end

    δ₁, δ₂, δ₃, δ₄
end

function update_lower_triangular_part_rq!(R, τ, k̄, b, γ, δ₁, δ₂, δ₃, δ₄)
    n, r = size(R.U)
    l,m = bandwidths(R.B)
    T = eltype(R)
    j = R.j[]
    #ū₁ = R.S[j+1, 1:r]
    ū₁ = view(R.S, j+1, 1:r)

    #η = -τ*γ[2:end]
    η = -τ*view(γ, 2:length(γ))
    #η[1:min(length(η),size(R.Φₛ,1)-1)] += R.Φₛ[2:min(length(η)+1,size(R.Φₛ,1)),:]*ū₁ - τ*R.Φₛ[2:min(length(η)+1,size(R.Φₛ,1)),:]*δ₂
    mul!(view(η, 1:min(length(η), size(R.Φₛ,1)-1)), view(R.Φₛ, 2:min(length(η)+1,size(R.Φₛ,1)),:), ū₁, one(T), one(T))
    mul!(view(η, 1:min(length(η), size(R.Φₛ,1)-1)), view(R.Φₛ, 2:min(length(η)+1,size(R.Φₛ,1)),:), δ₂, -τ, one(T))
    #η[1:length(δ₄)-1] += -τ*δ₄[2:end]
    view(η, 1:length(δ₄)-1) .+= -τ*view(δ₄, 2:length(δ₄))
    if l > 0
        #η[1:min(length(η), size(R.Λₛ,1)-1)] += R.Λₛ[2:min(length(η)+1,size(R.Λₛ,1)),1]
        view(η, 1:min(length(η), size(R.Λₛ,1)-1)) .+= view(R.Λₛ, 2:min(length(η)+1,size(R.Λₛ,1)), 1)
    end

    μ = -τ*k̄ + R.Ω*ū₁ - τ*δ₁ - τ*δ₃
    if l > 0
        #μ .+= R.Ψₛ[:,1]
        μ .+= view(R.Ψₛ, :, 1)
    end

    #R.B[j+2:j+1+length(η),j+1] = η 
    view(R.B, j+2:j+1+length(η), j+1) .= η 
    #R.V[j+1,:] = μ
    view(R.V, j+1, :) .= μ

end

function update_diagonal_rq!(R, τ, k̄, b, γ, δ₁, δ₂, δ₃, δ₄, r⁽¹⁾ᵀU⁽²⁾_table)
    j = R.j[]
    l,m = bandwidths(R.B)
    n, r = size(R.U)
    #ū₁ = R.S[j+1, 1:r]
    ū₁ = view(R.S, j+1, 1:r)
    #r⁽¹⁾ᵀU⁽²⁾ = (r⁽¹⁾ᵀU⁽²⁾_table[j+1, :])'
    r⁽¹⁾ᵀU⁽²⁾ = view(r⁽¹⁾ᵀU⁽²⁾_table, j+1, :)'
    diag_orig = R.B[j+1,j+1]
    R.B[j+1,j+1] = R.B[j+1,j+1] - τ*R.B[j+1,j+1] - τ*r⁽¹⁾ᵀU⁽²⁾*k̄ +
                   R.B[j+1,j+1]*(ū₁'*R.Ω*ū₁-τ*ū₁'*δ₁) + r⁽¹⁾ᵀU⁽²⁾*(R.Ω*ū₁-τ*δ₁) - 
                   τ*R.B[j+1,j+1]*ū₁'*δ₃ - τ*r⁽¹⁾ᵀU⁽²⁾*δ₃
    if l > 0
        #R.B[j+1,j+1] += -τ*γ[1] + R.Φₛ[1,:]'*ū₁ + diag_orig*ū₁'*R.Ψₛ[:,1] -
        #                τ*R.Φₛ[1,:]'*δ₂ + r⁽¹⁾ᵀU⁽²⁾*R.Ψₛ[:,1] - τ*δ₄[1] + R.Λₛ[1,1]
        R.B[j+1,j+1] += -τ*γ[1] + view(R.Φₛ, 1, :)'*ū₁ + diag_orig*ū₁'*view(R.Ψₛ, :, 1) -
                        τ*view(R.Φₛ, 1, :)'*δ₂ + r⁽¹⁾ᵀU⁽²⁾*view(R.Ψₛ, :, 1) - τ*δ₄[1] + R.Λₛ[1,1]
    end
end

function update_next_submatrix_rq!(R, τ, k̄, b, γ, δ₁, δ₂, δ₃, δ₄)
    T = eltype(R)

    #R.Ω .= -τ*k̄*k̄' + R.Ω - τ*δ₁*k̄' - τ*δ₃*k̄'
    mul!(R.Ω, k̄, k̄', -τ, one(T))
    mul!(R.Ω, δ₁, k̄', -τ, one(T))
    mul!(R.Ω, δ₃, k̄', -τ, one(T))

    temp_Ψₛ = R.Ψₛ[:,2:end] # need to copy the original R.Ψₛ first
    #R.Ψₛ[:,1:length(b)] = (-τ*k̄ - τ*δ₁ - τ*δ₃)*b'
    mul!(view(R.Ψₛ, :, 1:length(b)), k̄, b', -τ, zero(T))
    mul!(view(R.Ψₛ, :, 1:length(b)), δ₁, b', -τ, one(T))
    mul!(view(R.Ψₛ, :, 1:length(b)), δ₃, b', -τ, one(T))
    #R.Ψₛ[:,1:end-1] += temp_Ψₛ
    view(R.Ψₛ, :, 1:size(R.Ψₛ,2)-1) .+= temp_Ψₛ
    #R.Ψₛ[:,length(b)+1:end] .= 0
    view(R.Ψₛ, :, length(b)+1:size(R.Ψₛ,2)) .= zero(T)

    temp_Λₛ = R.Λₛ[2:end,2:end] # need to copy the original R.Λₛ first
    #R.Λₛ[1:length(γ)-1,1:length(b)] = -τ*γ[2:end]*b'
    mul!(view(R.Λₛ, 1:length(γ)-1, 1:length(b)), view(γ, 2:length(γ)), b', -τ, zero(T)) 
    #R.Λₛ[1:size(R.Φₛ,1)-1,1:length(b)] += -τ*R.Φₛ[2:end,:]*δ₂*b'
    mul!(view(R.Λₛ, 1:size(R.Φₛ,1)-1, 1:length(b)), view(R.Φₛ, 2:size(R.Φₛ,1), :), δ₂*b', -τ, one(T))
    #R.Λₛ[1:size(R.Λₛ,1)-1,1:size(R.Λₛ,2)-1] += temp_Λₛ
    view(R.Λₛ, 1:size(R.Λₛ,1)-1, 1:size(R.Λₛ,2)-1) .+= temp_Λₛ
    #R.Λₛ[1:length(δ₄)-1,1:length(b)] += -τ*δ₄[2:end]*b'
    mul!(view(R.Λₛ, 1:length(δ₄)-1, 1:length(b)), view(δ₄, 2:length(δ₄)), b', -τ, one(T))
    #R.Λₛ[length(γ):end,:] .= 0
    view(R.Λₛ, length(γ):size(R.Λₛ,1), :) .= zero(T)
    #R.Λₛ[:,length(b)+1:end] .= 0
    view(R.Λₛ, :,length(b)+1:size(R.Λₛ,2)) .= zero(T)

    #temp_Φₛ = R.Φₛ[2:end,:] - τ*R.Φₛ[2:end,:]*δ₂*k̄'
    temp_Φₛ = -τ*view(R.Φₛ, 2:size(R.Φₛ,1), :)*δ₂*k̄'
    temp_Φₛ .+= view(R.Φₛ, 2:size(R.Φₛ,1), :)
    #R.Φₛ[1:length(γ)-1,:] = -τ*γ[2:end]*k̄'
    mul!(view(R.Φₛ, 1:length(γ)-1, :), view(γ, 2:length(γ)), k̄', -τ, zero(T))
    #R.Φₛ[1:end-1,:] += temp_Φₛ
    view(R.Φₛ, 1:size(R.Φₛ,1)-1, :) .+= temp_Φₛ
    #R.Φₛ[1:length(δ₄)-1,:] += -τ*δ₄[2:end]*k̄'
    mul!(view(R.Φₛ, 1:length(δ₄)-1, :), view(δ₄, 2:length(δ₄)), k̄', -τ, one(T))
    #R.Φₛ[length(γ):end,1] .= 0
    view(R.Φₛ, length(γ):size(R.Φₛ,1), 1) .= zero(T)

end

function UᵀU_lookup_table_rq(R)
    n, r = size(R.U)
    T = eltype(R)
    UᵀU = zeros(T, n, r, r)
    UᵀU_current = zeros(T, r, r)
    for i in n:-1:1
        #UᵀU_current += R.S[i,1:r] * (R.S[i,1:r])'
        mul!(UᵀU_current, view(R.S, i, 1:r), view(R.S, i, 1:r)', one(T), one(T))
        #UᵀU[i,:,:] .= UᵀU_current
        view(UᵀU, i, :, :) .= UᵀU_current
    end
    UᵀU
end

function r⁽¹⁾ᵀU⁽²⁾_lookup_table(R)
    # R₀ = Matrix(triu(R.B) + triu(R.W*R.S', 1)) and r⁽¹⁾ᵀU⁽²⁾ = R₀[j+1,j+2:end]'*R.S[j+2:end,1:r]
    n, r = size(R.U)
    l, m = bandwidths(R.B)
    T = eltype(R)
    r⁽¹⁾ᵀU⁽²⁾ = zeros(T, n, r)
    #SᵀU = R.S' * R.S[:, 1:r]
    SᵀU = R.S' * view(R.S, :, 1:r)
    for i in 1:n-1
        #SᵀU -= R.S[i,:] * R.S[i,1:r]'
        mul!(SᵀU, view(R.S, i, :), view(R.S, i, 1:r)', -one(T), one(T))
        #r⁽¹⁾ᵀU⁽²⁾[i, :] = (R.W[i,:]' * SᵀU + R.B[i, i+1:min(i+m,n)]'*R.S[i+1:min(i+m,n),1:r])'
        mul!(view(r⁽¹⁾ᵀU⁽²⁾, i, :), SᵀU', view(R.W, i, :), one(T), zero(T))
        mul!(view(r⁽¹⁾ᵀU⁽²⁾, i, :), view(R.S, i+1:min(i+m,n),1:r)', view(R.B, i, i+1:min(i+m,n)), one(T), one(T)) 
    end
    r⁽¹⁾ᵀU⁽²⁾
end

function fast_RU(U, W, S, B, j)
    # compute R[j:end,j:end*U[j,end] where R = triu(B) + triu(WS',1) in O(n)
    n, r = size(U)
    p = size(W,2)
    l, m = bandwidths(B)
    T = eltype(U)
    RU = zeros(T, n+1-j, r)
    #SᵀU = S[j:n, :]' * S[j:n, 1:r]
    SᵀU = view(S, j:n, :)' * view(S, j:n, 1:r)
    for i in j:n
        #SᵀU -= S[i,:] * S[i,1:r]'
        mul!(SᵀU, view(S, i, :), view(S, i, 1:r)', -one(T), one(T))
        #RU[i+1-j,:] = (W[i,:]'*SᵀU + B[i,i:min(i+m,n)]'*S[i:min(i+m,n),1:r])'
        mul!(view(RU, i+1-j, :), SᵀU', view(W, i, :), one(T), zero(T))
        mul!(view(RU, i+1-j, :), view(S, i:min(i+m,n), 1:r)', view(B, i, i:min(i+m,n)), one(T), one(T))
    end
    RU
end

function symmetrize_from_lower!(B::BandedMatrix{T}) where T
    n = size(B,1)
    l, m = bandwidths(B)
    for i in 1:n
        for j in 1:min(m,n-i)
            B[i,i+j] = B[i+j,i]
        end
    end
end


###
# Support lmul!
###

function getproperty(F::QR{<:Any,<:BandedPlusSemiseparableMatrix}, d::Symbol)
    m, n = size(F)
    if d === :R
        return UpperTriangular(getfield(F, :factors))
    elseif d === :Q
        return QRPackedQ(getfield(F, :factors), F.τ)
    else
        getfield(F, d)
    end
end

function lmul!(adjQ::AdjointQ{<:Any,<:QRPackedQ{<:Any,<:BandedPlusSemiseparableMatrix}}, b::StridedVector)
    Q = parent(adjQ)
    F = Q.factors
    τ = Q.τ
    n, r = size(F.U)
    T = eltype(F.U)
    l, m = bandwidths(F.B)
    O = zeros(T, n, r)
    h = zeros(T, r)
    G = zeros(T, n, l+1)
    Uᵀb = Uᵀb_lookup_table(F, b)
    UᵀU = UᵀU_lookup_table(F)
    for j = 1 : n - 1
        # c is a scalar:
        #c = F.V[j,:]' * Uᵀb[j+1,:] + b[j] + F.B[j+1:min(j+l,n), j]' * b[j+1:min(j+l,n)] + F.V[j,:]'*UᵀU[j+1,:,:]*h + 
        #    (F.U[j,:]' + F.B[j+1:min(j+l,n), j]'*F.U[j+1:min(j+l,n),:])*h + sum(F.V[j,:]'*F.U[j+1:min(j+l,n),:]'*G[j+1:min(j+l,n),:]) + 
        #    sum(G[j,:]' + F.B[j+1:min(j+l,n), j]'*G[j+1:min(j+l,n),:])
        c = view(F.V, j, :)'*view(Uᵀb, j+1, :) + b[j] + view(F.B, j+1:min(j+l,n), j)'*view(b, j+1:min(j+l,n)) +
            view(F.V, j,:)'*view(UᵀU, j+1,:,:)*h + view(F.U, j,:)'*h + view(F.B, j+1:min(j+l,n),j)'*view(F.U, j+1:min(j+l,n),:)*h +
            sum(view(F.V,j,:)'*view(F.U,j+1:min(j+l,n),:)'*view(G,j+1:min(j+l,n),:)) + 
            sum(view(G,j,:)'+view(F.B,j+1:min(j+l,n), j)'*view(G,j+1:min(j+l,n),:))

        O[j, :] = F.U[j, :] .* h
        #h = h - τ[j] * c * F.V[j, :]
        mul!(h, I, view(F.V, j, :), -τ[j]*c, one(T))
        for t = 1 : min(l+1, n-j+1)
            if t == 1
                G[j+t-1, t] = -τ[j] * c * 1.0
            else
                G[j+t-1, t] = -τ[j] * c * F.B[j+t-1, j]
            end
        end
    end
    #b[1:end] += sum(O, dims=2) + sum(G, dims=2)
    for i = 1 : r
        mul!(view(b,:), I, view(O, :, i), one(T), one(T))
    end
    for i = 1 : l+1
        mul!(view(b,:), I, view(G, :, i), one(T), one(T))
    end
    #b[n] += F.U[n,:]'*h
    b[n] += view(F.U, n, :)' * h
end

function Uᵀb_lookup_table(F, b)
    n, r = size(F.U)
    T = eltype(F)
    Uᵀb = zeros(T, n, r)
    Uᵀb_current = zeros(T, r)
    for i in n:-1:1
        #Uᵀb_current .+= F.U[i,:] * b[i]
        mul!(Uᵀb_current, I, view(F.U, i, :), b[i], one(T))
        #Uᵀb[i,:] .= Uᵀb_current
        view(Uᵀb, i, :) .= Uᵀb_current
    end
    Uᵀb
end