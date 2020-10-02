using SemiseparableMatrices, LazyArrays, Test

@testset "Inverse of banded" begin
    @testset "Bidiagonal" begin
        @testset "SemiseparableMatrix" begin
            n = 5
            L = Bidiagonal(randn(n), randn(n-1), :L)
            U = Bidiagonal(randn(n), randn(n-1), :U)
            Li = SemiseparableMatrix(ApplyArray(inv, L))
            Ui = SemiseparableMatrix(ApplyArray(inv, U))
            @test Li ≈ inv(L)
            @test Ui ≈ inv(U)
        end

        @testset "Derivation" begin
            n = 5
            L = Bidiagonal(ones(n), randn(n-1), :L)
            Li = ApplyArray(inv, L)
            ℓ = [1; cumprod(-L.ev)]
            @test tril(ℓ * inv.(ℓ)') ≈ Li

            L = Bidiagonal(randn(n), randn(n-1), :L)
            L̃ = Bidiagonal(ones(n), L.ev ./ L.dv[1:end-1], :L)
            # inv(L) == inv(L̃ * D) == inv(D) * inv(L̃)
            ℓ = [1; cumprod(-L.ev./L.dv[1:end-1])]
            @test tril(ℓ * inv.(ℓ)') ≈ inv(L̃)
            @test tril((inv.(L.dv) .* ℓ) * inv.(ℓ)') ≈ inv(L)

            n = 5
            U = Bidiagonal(ones(n), randn(n-1), :U)
            u = [1; cumprod(-U.ev)]
            @test triu(inv.(u) * u') ≈ inv(U)

            U = Bidiagonal(randn(n), randn(n-1), :U)
            u = [1; cumprod(-U.ev./U.dv[1:end-1])]
            @test triu(inv.(u) * (inv.(U.dv) .* u)') ≈ inv(U)
        end
    end
end