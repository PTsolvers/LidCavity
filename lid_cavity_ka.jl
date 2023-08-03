using GLMakie
# Makie.inline!(true)

using KernelAbstractions

macro isin(A) esc(:(checkbounds(Bool, $A, ix, iy))) end

macro all(A) esc(:($A[ix, iy])) end

macro inn(A) esc(:($A[ix+1, iy+1])) end

macro ∂_x(A) esc(:($A[ix+1, iy] - $A[ix, iy])) end
macro ∂_y(A) esc(:($A[ix, iy+1] - $A[ix, iy])) end

macro ∂_xi(A) esc(:($A[ix+1, iy+1] - $A[ix, iy+1])) end
macro ∂_yi(A) esc(:($A[ix+1, iy+1] - $A[ix+1, iy])) end

@kernel function update_σ!(Pr, τ, V, μ, Δτ, Δ)
    ix, iy = @index(Global, NTuple)
    @inbounds if isin(Pr)
        exx = @∂_xi(V.x) / Δ.x
        eyy = @∂_yi(V.y) / Δ.y
        ∇V = exx + eyy
        @all(Pr) -= ∇V * Δτ.Pr
        @all(τ.xx) += (-@all(τ.xx) + 2.0 * μ * (exx - ∇V / 3.0)) * Δτ.τ
        @all(τ.yy) += (-@all(τ.yy) + 2.0 * μ * (eyy - ∇V / 3.0)) * Δτ.τ
    end
    @inbounds if isin(τ.xy)
        exy = 0.5 * (@∂_y(V.x) / Δ.y + @∂_x(V.y) / Δ.x)
        @all(τ.xy) += (-@all(τ.xy) + 2.0 * exy) * Δτ.τ
    end
end

#   ----
# |-x-|-x-|-x-|
#       ⬆
#      -->

# |-x-|-x-|-x-|
#     ^ ⬆
#     |
#    -->

# rVx = dt * (max(Vy[1:end-1], 0.0) * d(Vx[1:end-1])/dy + min(Vy[2:end], 0.0) * d(Vx[2:end  ])/dy)

@kernel function update_rV!(rV, V, Pr, τ, ρ, Δ)
    ix, iy = @index(Global, NTuple)
    @inbounds if isin(rV.x)
        V∇Vx = max(0.0, V.x[ix+1, iy+1]) * (V.x[ix+1, iy+1] - V.x[ix  , iy+1]) / Δ.x +
               min(0.0, V.x[ix+1, iy+1]) * (V.x[ix+2, iy+1] - V.x[ix+1, iy+1]) / Δ.x +
               max(0.0, 0.5 * (V.y[ix+1, iy  ] + V.y[ix+2, iy  ])) * (V.x[ix+1, iy+1] - V.x[ix+1, iy  ]) / Δ.y +
               min(0.0, 0.5 * (V.y[ix+1, iy+1] + V.y[ix+2, iy+1])) * (V.x[ix+1, iy+2] - V.x[ix+1, iy+1]) / Δ.y
        
        @all(rV.x) = -@∂_x(Pr) / Δ.x + @∂_x(τ.xx) / Δ.x + @∂_yi(τ.xy) / Δ.y - ρ * V∇Vx
    end
            
    @inbounds if isin(rV.x)
        V∇Vy = max(0.0, V.y[ix+1, iy+1]) * (V.y[ix+1, iy+1] - V.y[ix  , iy+1]) / Δ.y +
               min(0.0, V.y[ix+1, iy+1]) * (V.y[ix+1, iy+2] - V.y[ix+1, iy+1]) / Δ.y +
               max(0.0, 0.5 * (V.x[ix  , iy+1] + V.x[ix  , iy+2])) * (V.y[ix+1, iy+1] - V.y[ix  , iy+1]) / Δ.x +
               min(0.0, 0.5 * (V.x[ix+1, iy+1] + V.x[ix+1, iy+2])) * (V.y[ix+2, iy+1] - V.y[ix+1, iy+1]) / Δ.x

        @all(rV.y) = -@∂_y(Pr) / Δ.y + @∂_y(τ.yy) / Δ.y + @∂_xi(τ.xy) / Δ.x - ρ * V∇Vy
    end
end

@kernel function update_V!(V, rV, Δτ)
    ix, iy = @index(Global, NTuple)
    @inbounds if isin(rV.x) @inn(V.x) += @all(rV.x) * Δτ.V end
    @inbounds if isin(rV.y) @inn(V.y) += @all(rV.y) * Δτ.V end
end

@kernel function bc_Vx!(Vx, U)
    iy = @index(Global, Linear)
    @inbounds Vx[1, iy] = 2.0 * U - Vx[2    , iy]
    @inbounds Vx[2, iy] = 2.0 * U - Vx[end-1, iy]
end

@kernel function bc_Vy!(Vy, U)
    ix = @index(Global, Linear)
    @inbounds Vy[ix, 1  ] = 2.0 * U - Vy[ix, 2    ]
    @inbounds Vy[ix, end] = 2.0 * U - Vy[ix, end-1]
end

@views amean1(A) = 0.5 .* (A[1:end-1] .+ A[2:end])
@views avx(A) = 0.5 .* (A[1:end-1, :] .+ A[2:end, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])

@views function stokes(backend = CPU())
    ka_zeros(sz...) = KernelAbstractions.zeros(backend, sz...)
    # Physics
    h       = 1.0
    lx = ly = h
    μ       = 1.0
    U       = 1.0
    ρ       = 100.0 # Re = ρ * U * ly / μs
    # Numerics
    nx      = ny = 100
    ndt     = 1000
    niter   = 3e7
    ϵ       = 1e-5
    r       = 0.7
    re_mech = 20π
    # Preprocessing
    dx, dy  = lx / nx, ly / ny
    xv, yv  = LinRange(-lx / 2, lx / 2, nx + 1), LinRange(0, ly, ny + 1)
    xc, yc  = amean1(xv), amean1(yv)
    lτ_re_m = min(lx, ly) / re_mech
    vdτ     = min(dx, dy) / sqrt(6.1)
    θ_dτ    = lτ_re_m * (r + 4 / 3) / vdτ
    dτ_r    = 1.0 ./ (θ_dτ .+ 1.0)
    nudτ    = vdτ * lτ_re_m
    dτ_Q    = min(dx, dy)^2 / 4.1
    Δτ = (
        Pr = r * μ / θ_dτ,
        τ = dτ_r,
        V = nudτ / μ,
    )
    @info "Here -1"
    # Initialisation
    Pr      = ka_zeros(nx, ny)
    V       = (
        x = ka_zeros(nx + 1, ny + 2),
        y = ka_zeros(nx + 2, ny + 1),
    )
    τ   = (
        xx = ka_zeros(nx, ny),
        yy = ka_zeros(nx, ny),
        xy = ka_zeros(nx + 1, ny + 1),
    )
    rV =  (
        x = ka_zeros(nx - 1, ny),
        y = ka_zeros(nx, ny - 1),
    )
    Q  = ka_zeros(nx + 1, ny + 1)
    UV = ka_zeros(nx + 1, ny + 1)
    RQ = ka_zeros(nx - 1, ny - 1)
    @info "Here 0"

    bc_Vx!(backend, 256, size(V.x, 2))(V.x, 0.0)
    bc_Vy!(backend, 256, size(V.y, 1))(V.y, U)
    KernelAbstractions.synchronize(backend)
    # Action
    iter = 0
    res = ϵ * 2
    while (max(res) > ϵ) && (iter < niter)
        # # Pressure
        # ∇v .= diff(Vx, dims=1) ./ dx + diff(Vy, dims=2) ./ dy
        # Pr .-= ∇v ./ (1.0 / (r * μ / θ_dτ))
        # # Velocity + viscous rheology
        # Vxe .= hcat(.-Vx[:, 1], Vx, 2.0 .* U .- Vx[:, end])
        # Vye .= vcat(.-Vy[1, :]', Vy, .-Vy[end, :]')
        # εxx .= diff(Vx, dims=1) ./ dx .- ∇v ./ 3.0
        # εyy .= diff(Vy, dims=2) ./ dy .- ∇v ./ 3.0
        # εxyv .= 0.5 .* (diff(Vxe, dims=2) ./ dy .+ diff(Vye, dims=1) ./ dx)
        # τxx .+= (.-τxx .+ 2.0 .* μ .* εxx) .* dτ_r
        # τyy .+= (.-τyy .+ 2.0 .* μ .* εyy) .* dτ_r
        # τxyv .+= (.-τxyv .+ 2.0 .* μ .* εxyv) .* dτ_r
        # Rx .= diff(.-Pr .+ τxx, dims=1) ./ dx .+ diff(τxyv[2:end-1, :], dims=2) ./ dy
        # Ry .= diff(.-Pr .+ τyy, dims=2) ./ dy .+ diff(τxyv[:, 2:end-1], dims=1) ./ dx
        # # Advection
        # dVx .= max.(0.0, Vx[2:end-1, 2:end-1]) .* diff(Vx[1:end-1, 2:end-1], dims=1) ./ dx
        # dVx .+= min.(0.0, Vx[2:end-1, 2:end-1]) .* diff(Vx[2:end, 2:end-1], dims=1) ./ dx
        # dVx .+= max.(0.0, avx(Vy[:, 2:end-2])) .* diff(Vx[2:end-1, 1:end-1], dims=2) ./ dy
        # dVx .+= min.(0.0, avx(Vy[:, 3:end-1])) .* diff(Vx[2:end-1, 2:end], dims=2) ./ dy
        # dVy .= max.(0.0, Vy[2:end-1, 2:end-1]) .* diff(Vy[2:end-1, 1:end-1], dims=2) ./ dy
        # dVy .+= min.(0.0, Vy[2:end-1, 2:end-1]) .* diff(Vy[2:end-1, 2:end], dims=2) ./ dy
        # dVy .+= max.(0.0, avy(Vx[2:end-2, :])) .* diff(Vy[1:end-1, 2:end-1], dims=1) ./ dx
        # dVy .+= min.(0.0, avy(Vx[3:end-1, :])) .* diff(Vy[2:end, 2:end-1], dims=1) ./ dx
        # Rx[:, 2:end-1] .-= ρ .* dVx
        # Ry[2:end-1, :] .-= ρ .* dVy
        # # Update
        # Vx[2:end-1, :] .+= Rx .* nudτ ./ μ
        # Vy[:, 2:end-1] .+= Ry .* nudτ ./ μ
        # Stream function
        # UV .= diff(Vxe, dims=2) ./ dy .- diff(Vye, dims=1) ./ dx
        # RQ .= .-(diff(diff(Q[2:end-1, :], dims=2), dims=2) ./ dy^2 .+
        #          diff(diff(Q[:, 2:end-1], dims=1), dims=1) ./ dx^2) .+ UV[2:end-1, 2:end-1] .+ (1 - 2 / nx) * RQ
        # Q[2:end-1, 2:end-1] .-= dτ_Q * RQ

        @info "Here 1"
        update_σ!(backend, 256, (nx+1,ny+1))(Pr, τ, V, μ, Δτ, Δ)
        @info "Here 2"
        update_rV!(backend, 256, (nx, ny))(rV, V, Pr, τ, ρ, Δ)
        @info "Here 3"
        update_V!(backend, 256, (nx, ny))(V, rV, Δτ)
        @info "Here 4"
        bc_Vx!(backend, 256, size(V.x, 2))(V.x, 0.0)
        bc_Vy!(backend, 256, size(V.y, 1))(V.y, U)

        KernelAbstractions.synchronize(backend)

        if iter % ndt == 0
            resv  = maximum.((Rx, Ry, ∇v, RQ))
            res   = maximum(resv); println(" iter $iter err: $(round.(resv, sigdigits=3))")
        end
        iter += 1
    end
    # visualise
    fig, ax, hm = contourf(xc, yc, avy(Vy); levels=20, figure=(resolution=(1000, 800), fontsize=30), axis=(aspect=DataAspect(), title="Velocity"), colormap=:jet)
    contour!(ax, xv[2:end-1], yv[2:end-1], log10.(abs.(Q[2:end-1, 2:end-1])); levels=18, color=:black)
    Colorbar(fig[:, end+1], hm)
    limits!(ax, -lx / 2, lx / 2, 0, ly)
    display(fig)
    return
end

stokes()

# fig, ax, hm = contourf(xv, yv, Q; levels=20, figure=(resolution=(1000, 800), fontsize=30), axis=(aspect=DataAspect(), title="Stream function"), colormap=:jet)