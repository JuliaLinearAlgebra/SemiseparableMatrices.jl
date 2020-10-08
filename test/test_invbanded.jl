using SemiseparableMatrices, LazyArrays, MatrixFactorizations, Test

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

        @testset "Tridiagonal inverse" begin
            n = 10
            A = Tridiagonal(randn(n-1), randn(n), randn(n-1)) + 10I
            U,L = ul(A, Val(false))
            Li = SemiseparableMatrix(ApplyArray(inv, L))
            Ui = SemiseparableMatrix(ApplyArray(inv, U))
            Ai = inv(A)
            @test Li * Ui ≈ Ai
            @test rank(triu(Ai)[1:5,5:end]) == 1
            a,b = Li.L.args
            c,d = Ui.U.args
            @test Ai[1,1] ≈ a[1]b[1]c[1]d[1]
            @test Ai[1,2] ≈ a[1]b[1]c[1]d[2]
            @test Ai[1,3] ≈ a[1]b[1]c[1]d[3]
            @test Ai[1,5] ≈ a[1]b[1]c[1]d[5]
            @test Ai[2,1] ≈ a[2]b[1]c[1]d[1]
            @test Ai[2,2] ≈ a[2]b[1]c[1]d[2]+a[2]b[2]c[2]d[2]
            @test Ai[2,3] ≈ a[2]b[1]c[1]d[3]+a[2]b[2]c[2]d[3]
            @test Ai[3,1] ≈ a[3]b[1]c[1]d[1]
            @test Ai[3,2] ≈ a[3]b[1]c[1]d[2]+a[3]b[2]c[2]d[2]
            @test Ai[3,3] ≈ a[3]b[1]c[1]d[3]+a[3]b[2]c[2]d[3]+a[3]b[3]c[3]d[3]
            @test Ai[3,4] ≈ a[3]b[1]c[1]d[4]+a[3]b[2]c[2]d[4]+a[3]b[3]c[3]d[4]

            @test Ai[1:3,3:5] ≈ [a[1]b[1]c[1]; a[2]*(b[1]c[1]+b[2]c[2]); a[3]*(b[1]c[1]+b[2]c[2]+b[3]c[3])]*d[:,3:5]
            @test Ai[3:5,1:3] ≈ a[3:5] * [b[1]c[1]d[1] (b[1]c[1]+b[2]c[2])*d[2] (b[1]c[1]+b[2]c[2]+b[3]c[3])*d[3]]

            b̃ = Array{Float64}(undef, 1, n)
            for j = 1:n
                b̃[1,j] = 0
                for κ = 1:j
                    b̃[1,j] += b[κ]c[κ]
                end
                b̃[1,j] *= d[j]
            end
            c̃ = Array{Float64}(undef, n)
            for j = 1:n
                c̃[j] = 0
                for κ = 1:j
                    c̃[j] += b[κ]c[κ]
                end
                c̃[j] *= a[j]
            end
            @test tril(a*b̃) ≈ tril(Ai)
            @test triu(c̃*d) ≈ triu(Ai)
        end
    end
end