###
# Here we implement inversion of banded matrices, beginning with Bidiagonal
###

function SemiseparableMatrix(Bi::ApplyMatrix{T,typeof(inv),<:Tuple{Bidiagonal}}) where T
    B = inv(Bi)
    n = size(B,1)
    ℓ = [1; cumprod(-B.ev./B.dv[1:end-1])]
    if B.uplo == 'L'
        SemiseparableMatrix(ApplyArray(*, inv.(B.dv) .* ℓ, inv.(ℓ)'), Zeros{T}(n,n), 0, 1)
    else
        SemiseparableMatrix(Zeros{T}(n,n), ApplyArray(*, inv.(ℓ), (inv.(B.dv) .* ℓ)'), 1, 0)
    end
end