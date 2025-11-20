struct BandedPlusSemiseparableMatrix{T} <: LayoutMatrix{T}
    # representing B + tril(UV', -1) + triu(WS', 1)
    B::BandedMatrix{T} 
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

"""
Represents factors matrix for QR for banded+semiseparable. we have

    for j = 0, we have A₀=tril(UV',-1) + B + triu(WS',1);
    F = qrfactUnblocked!(A₀).factors # full QR factors;
    As j increases,  for BandedPlusSemiseparableQRPerturbedFactors A:
    A[1:j,:] == F[1:j,:];
    A[:,1:j] == F[:,1:j];
    A[k, k] == F[k, k] for k < j;
    A[j+1:end,j+1:end] == A₀[j+1:end,j+1:end] + U[j+1:end,:]*Q*S[j+1:end,1:p]' + U[j+1:end,:]*K*U[j+1:end]'*A₀[j+1:end,j+1:end]
                          + U[j+1:end,:]*[Eₛ 0] + [Xₛ;0]*S[j+1:end,1:p]' + [Yₛ;0]*U[j+1:end]'*A₀[j+1:end,j+1:end] + [Zₛ 0;0 0]
"""

struct BandedPlusSemiseparableQRPerturbedFactors{T} <: LayoutMatrix{T}
    B::BandedMatrix{T} # lower bandwidth l and upper bandwidth l + m
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
        A = tril(U*V',-1) + B + triu(W*S',1)
        AᵀU = (fast_UᵀA(U, V, W, S, B, 1))'
        BandedPlusSemiseparableQRPerturbedFactors(BandedMatrix(B,(l,l+m)),U,V,[W zeros(n,r)],[S AᵀU],
            zeros(r,p),zeros(r,r),zeros(r,min(l+m,n)),zeros(min(l,n),p),zeros(min(l,n),r),zeros(min(l,n),min(l+m,n)),Ref(0))
    else
        throw(ErrorException("Dimensions not match!"))
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

    bandedplussemi_qr!(A,  zeros(T,size(A.B, 1)), UᵀU_lookup_table(A), ūw̄_sum_lookup_table(A), d_extra_lookup_table(A))
end

function bandedplussemi_qr!(A, τ, tables...)
    n = size(A.B, 1)
    for i in 1 : n-1
        onestep_qr!(A, τ, tables...)
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
    k̄, b, pivot_new, τ_current = compute_Householder_vector(A, UᵀU) # I- τyy' where y = eⱼ₊₁ +  U⁽ʲ⁺²⁾k̄ + b and p_new will be the diagonal element on the current column
    τ[A.j[]+1] = τ_current
    w̄₁, ū₁, d₁, f, d̄ = compute_d₁_and_d̄(A, b)
    c₁, c₂, c₃, c₄, c₅, c₆ = compute_variables_c(A, k̄, b, UᵀU)

    Q_prev = copy(A.Q)
    K_prev = copy(A.K)
    Eₛ_prev = copy(A.Eₛ)
    Xₛ_prev = copy(A.Xₛ)
    Yₛ_prev = copy(A.Yₛ)
    Zₛ_prev = copy(A.Zₛ)
    
    update_next_submatrix!(A, k̄, b, τ_current, w̄₁, ū₁, d₁, f, d̄, c₁, c₂, c₃, c₄, c₅, c₆, Q_prev, K_prev, Eₛ_prev, Xₛ_prev, Yₛ_prev, Zₛ_prev)
    update_upper_triangular_part!(A, k̄, b, τ_current, w̄₁, ū₁, d₁, f, d̄, c₁, c₂, c₃, c₄, c₅, c₆, Q_prev, K_prev, Eₛ_prev, Xₛ_prev, Yₛ_prev, Zₛ_prev, ūw̄_sum, d_extra)
    update_lower_triangular_part!(A, pivot_new, k̄, b)

    A.j[] += 1

end

# the following are auxiliary functions:

function compute_Householder_vector(A, UᵀU)
    # compute Householder transformation I- τyy' where y = eⱼ₊₁ +  U⁽ʲ⁺²⁾k̄ + b

    # first express A[j+1:end,j+1] as p*eⱼ₊₁ + U⁽ʲ⁺²⁾k̄ + b
    j = A.j[]
    n = size(A.B, 1)
    p = size(A.W, 2) - size(A.U, 2) # W already been padded to have size n × (p+r)
    l, m = bandwidths(A.B)
    m = m - l # B already been padded to have upper bandwidth l+m
    UᵀA_1 = (view(A.U, j+1:min(l+j+1, n),:))' * view(A.B, j+1:min(l+j+1, n) , j+1) + view(UᵀU, j+2 , : , :) * view(A.V, j+1, :) #the j+1th column of UᵀA

    k̄ = view(A.V, j+1, :) + A.Q * view(A.S, j+1, 1:p) + A.K * UᵀA_1
    if l > 0 || m > 0
        k̄ .+= view(A.Eₛ, :, 1)
    end
    b = A.B[j+2:min(l+j+1, n), j+1]
    if l > 0 || m > 0
        b[1:min(l-1,length(b))] .+= view((A.Xₛ * view(A.S, j+1, 1:p) + A.Yₛ * UᵀA_1 + view(A.Zₛ, :, 1)), 2:min(l,length(b)+1))
    end
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
    n = size(A.B, 1)
    p = size(A.W, 2) - size(A.U, 2) # W already been padded to have size n × (p+r)
    l, m = bandwidths(A.B)
    m = m - l # B already been padded to have upper bandwidth l+m
    w̄₁ = view(A.W, j+1, 1:p)
    ū₁ = view(A.U, j+1, :)
    d₁ = A.B[j+1,j+1:min(j+1+m, n)]
    d₁[1] = d₁[1] - w̄₁'*(view(A.S, j+1, 1:p))
    f = (view(A.W, j+2:j+1+length(b), 1:p))' * b
    index1 = j+2:j+1+length(b)
    index2 = j+1:min(j+1+m+l, n)
    tril_sub = [ (ii - jj) >= 1 for ii in index1, jj in index2 ]
    triu_sub = [ (jj - ii) >= 1 for ii in index1, jj in index2 ]
    A₀ = view(A.U, index1, :) * view(A.V, index2, :)' .* tril_sub + view(A.B, index1, index2) + view(A.W, index1, 1:p) * view(A.S, index2, 1:p)' .* triu_sub
    d̄ = (b'A₀ - f'*(view(A.S, index2, 1:p))')'
    w̄₁, ū₁, d₁, f, d̄
end

function compute_variables_c(A, k̄, b, UᵀU)
    j = A.j[]
    n = size(A.B, 1)
    l, m = bandwidths(A.B)
    m = m - l # B already been padded to have upper bandwidth l+m
    UᵀU⁽²⁾ = view(UᵀU, j+2, :, :)

    c₁ = (A.Q)' * view(A.U, j+1, :) + (A.Q)' * UᵀU⁽²⁾ * k̄ + (A.Q)' * (view(A.U, j+2:j+1+length(b), :))' * b
    c₂ = (A.K)' * view(A.U, j+1, :) + (A.K)' * UᵀU⁽²⁾ * k̄ + (A.K)' * (view(A.U, j+2:j+1+length(b), :))' * b
    c₃ = view(A.U, j+1, :) + UᵀU⁽²⁾ * k̄ + (view(A.U, j+2:j+1+length(b), :))' * b

    Xₛ_valid = view(A.Xₛ, 2:min(l, n-j), :)
    Yₛ_valid = view(A.Yₛ, 2:min(l, n-j), :)
    Zₛ_valid = view(A.Zₛ, 2:min(l, n-j), 1:min(l+m, n-j))
    c₄ = (Xₛ_valid)' * view(A.U, j+2:j+1+size(Xₛ_valid,1), :) * k̄ + (Xₛ_valid)' * view(b, 1:size(Xₛ_valid,1))
    c₅ = (Yₛ_valid)' * view(A.U, j+2:j+1+size(Yₛ_valid,1), :) * k̄ + (Yₛ_valid)' * view(b, 1:size(Yₛ_valid,1))
    c₆ = (Zₛ_valid)' * view(A.U, j+2:j+1+size(Zₛ_valid,1), :) * k̄ + (Zₛ_valid)' * view(b, 1:size(Zₛ_valid,1))
    if l > 0
        c₄ .+= view(A.Xₛ, 1, :)
        c₅ .+= view(A.Yₛ, 1, :)
        c₆ .+= view(A.Zₛ, 1, 1:min(l+m, n-j))
    end

    c₁, c₂, c₃, c₄, c₅, c₆
end

function update_next_submatrix!(A, k̄, b, τ, w̄₁, ū₁, d₁, f, d̄, c₁, c₂, c₃, c₄, c₅, c₆, Q_prev, K_prev, Eₛ_prev, Xₛ_prev, Yₛ_prev, Zₛ_prev)  
    A.Q[:,:] = -τ*k̄*w̄₁' - τ*k̄*f' + Q_prev - τ*k̄*c₁' + K_prev*ū₁*w̄₁'-
                τ*k̄*c₂'*ū₁*w̄₁' - τ*k̄*c₄' - τ*k̄*c₅'*ū₁*w̄₁'
    A.K[:,:] = -τ*k̄*k̄' + K_prev - τ*k̄*c₂' - τ*k̄*c₅'

    A.Eₛ[:,1:length(d̄)-1] = -τ*k̄*(view(d̄, 2:length(d̄)))'
    A.Eₛ[:,1:length(d₁)-1] .+= (-τ*k̄ + K_prev*ū₁ - τ*k̄*c₂'*ū₁ - τ*k̄*c₅'*ū₁)*(view(d₁, 2:length(d₁)))'
    A.Eₛ[:,1:end-1] .+= view(Eₛ_prev, :, 2:size(Eₛ_prev,2)) - τ*k̄*c₃'*view(Eₛ_prev, :, 2:size(Eₛ_prev,2))
    A.Eₛ[:,1:length(c₆)-1] .+= -τ*k̄*(view(c₆, 2:length(c₆)))'
    A.Eₛ[:,length(d̄):end] .= zero(eltype(A))

    A.Xₛ[1:length(b),:] = b*(-τ*w̄₁' - τ*f' - τ*c₁' - τ*c₂'*ū₁*w̄₁' - τ*c₄' - τ*c₅'*ū₁*w̄₁')
    A.Xₛ[1:end-1,:] .+= view(Xₛ_prev, 2:size(Xₛ_prev,1), :) + view(Yₛ_prev, 2:size(Yₛ_prev,1), :)*ū₁*w̄₁'
    A.Xₛ[length(b)+1:end,:] .= zero(eltype(A))

    A.Yₛ[1:length(b),:] = b*(-τ*k̄' - τ*c₂' - τ*c₅')
    A.Yₛ[1:end-1,:] .+= view(Yₛ_prev, 2:size(Yₛ_prev,1), :)
    A.Yₛ[length(b)+1:end,:] .= zero(eltype(A))

    A.Zₛ[1:length(b), 1:length(d̄)-1] = -τ*b*(view(d̄, 2:length(d̄)))'
    A.Zₛ[1:length(b), 1:length(d₁)-1] .+= b*(-τ - τ*c₂'*ū₁ - τ*c₅'*ū₁)*(view(d₁, 2:length(d₁)))'
    A.Zₛ[1:length(b), 1:end-1] .+= -τ*b*c₃'*view(Eₛ_prev, :, 2:size(Eₛ_prev, 2))
    A.Zₛ[1:end-1, 1:length(d₁)-1] .+= view(Yₛ_prev, 2:size(Yₛ_prev,1), :)*ū₁*(view(d₁, 2:length(d₁)))'
    A.Zₛ[1:end-1, 1:end-1] .+= view(Zₛ_prev, 2:size(Zₛ_prev,1), 2:size(Zₛ_prev,2))
    A.Zₛ[1:length(b), 1:length(c₆)-1] .+= -τ*b*(view(c₆, 2:length(c₆)))'
    A.Zₛ[length(b)+1:end,:] .= zero(eltype(A))
    A.Zₛ[:,length(d̄):end] .= zero(eltype(A))

end

function update_upper_triangular_part!(A, k̄, b, τ, w̄₁, ū₁, d₁, f, d̄, c₁, c₂, c₃, c₄, c₅, c₆, Q, K, Eₛ, Xₛ, Yₛ, Zₛ, ūw̄_sum, d_extra)
    j = A.j[]
    p = size(A.W, 2) - size(A.U, 2) # W already been padded to have size n × (p+r)
    β = (-τ*k̄' + ū₁'*K - τ*c₂' - τ*c₅')'
    if size(Yₛ, 1) > 0
        β .+= view(Yₛ, 1, :)
    end

    α = (w̄₁' - τ*w̄₁' - τ*f' + ū₁'*Q - τ*c₁' - τ*c₄' + τ*k̄'*ū₁*w̄₁')'
    if size(Xₛ, 1) > 0
        α .+= view(Xₛ, 1, :)
    end
    α .+= (-β' * view(ūw̄_sum, j+1, :, :))'

    d = -τ * d̄[2:end]
    d[1:length(d₁)-1] .+= (1 - τ + τ*k̄'*ū₁) * view(d₁, 2:length(d₁))
    d[1:min(length(d),size(Eₛ,2)-1)] .+= ((ū₁' - τ*c₃') * view(Eₛ, :, 2:min(length(d)+1,size(Eₛ,2))))'
    d[1:length(c₆)-1] .+= -τ * view(c₆, 2:length(c₆))
    if size(Zₛ, 1) > 0
        d[1:min(length(d),size(Zₛ,2)-1)] .+= view(Zₛ, 1, 2:min(length(d)+1,size(Zₛ,2)))
    end
    d_extra_current = d_extra[j+1]
    d[1:size(d_extra_current,2)] .+= (-β'*d_extra_current)'

    j = A.j[]
    A.W[j+1, 1:p] = α
    A.W[j+1, p+1:end] = β
    A.B[j+1, j+2:j+1+length(d)] = d
end

function update_lower_triangular_part!(A, pivot, k̄, b)
    j = A.j[]
    A.B[j+1, j+1] = pivot
    A.B[j+2:j+1+length(b), j+1] = b
    A.V[j+1, :] = k̄
end

function UᵀU_lookup_table(A)
    n, r = size(A.U) 
    UᵀU = zeros(eltype(A), n, r, r)
    UᵀU_current = zeros(eltype(A), r, r)
    for i in n:-1:1
        UᵀU_current .+= view(A.U, i, :) * (view(A.U, i, :))'
        UᵀU[i,:,:] .= UᵀU_current
    end
    UᵀU
end

function ūw̄_sum_lookup_table(A)
    n, r = size(A.U)
    p = size(A.W, 2) - size(A.U, 2) # W already been padded to have size n × (p+r)
    ūw̄_sum = zeros(eltype(A), n, r, p)
    ūw̄_sum_current = zeros(eltype(A), r, p)
    for t in 1:n
        ūw̄_sum[t,:,:] .= ūw̄_sum_current
        ūw̄_sum_current .+= view(A.U, t, :) * (view(A.W, t, 1:p))'
    end
    ūw̄_sum
end

function d_extra_lookup_table(A)
    n, r = size(A.U)
    l, m = bandwidths(A.B)
    m = m - l # B already been padded to have upper bandwidth l+m
    d_extra = Vector{Matrix{eltype(A)}}()
    for i in 1:n
        d_extra_current = zeros(eltype(A), r, min(m, n-i))
        for t in max(1,i+1-m) : i-1
            d_extra_current .+= view(A.U, t, :) * (view(A.B, t, i+1:i+size(d_extra_current,2)))'
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
    UᵀA = zeros(eltype(U), r, n+1-j)
    UᵀU = (view(U, j:n, :))' * view(U, j:n, :)
    UᵀW = zeros(eltype(U), r, p)
    for i in j:n
        UᵀU .-= view(U, i, :) * (view(U, i, :))'
        UᵀA[:,i+1-j] = UᵀU*view(V, i, :) + (view(U, max(j,i-m) : min(i+l,n), :))'*view(B, max(j,i-m) : min(i+l,n), i) + UᵀW*view(S, i, :)
        UᵀW .+= view(U, i, :) * (view(W, i, :))'
    end
    UᵀA
end