using CairoMakie
using AMDGPU
using KernelAbstractions
const KA = KernelAbstractions

@views amean1(A) = 0.5 .* (A[1:end-1] .+ A[2:end])
@views avx(A)    = 0.5 .* (A[1:end-1, :] .+ A[2:end, :])
@views avy(A)    = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])
inn_x(A)  = @view A[2:end-1,:]
inn_y(A)  = @view A[:,2:end-1]

macro isin(A) esc(:( checkbounds(Bool, $A, ix, iy) )) end
macro ∂_∂x(A) esc(:( ($A[ix+1, iy] - $A[ix, iy]) )) end
macro ∂_∂y(A) esc(:( ($A[ix, iy+1] - $A[ix, iy]) )) end

@kernel function update_Pr!(∇v, Pr, Vxe, Vye, Vx, Vy, r, μs, θ_dτ, dx, dy)
    ix, iy = @index(Global, NTuple)
    if @isin(∇v) @inbounds ∇v[ix, iy]  = @∂_∂x(Vx) / dx + @∂_∂y(Vy) / dy end
    if @isin(Pr) @inbounds Pr[ix, iy] -= ∇v[ix, iy] / (1.0 / (r * μs / θ_dτ)) end
    if @isin(Pr) @inbounds Vxe[ix, iy] = Vx[ix, iy] end
    if @isin(Pr) @inbounds Vye[ix, iy] = Vy[ix, iy] end
end

@kernel function bc_Vx!(Vxe, U1, U2)
    iy = @index(Global, Linear)
    @inbounds Vxe[1  , iy] = 2.0 * U1 - Vxe[2    , iy]
    @inbounds Vxe[end, iy] = 2.0 * U2 - Vxe[end-1, iy]
end

@kernel function bc_Vy!(Vye, U1, U2)
    ix = @index(Global, Linear)
    @inbounds Vye[ix, 1  ] = 2.0 * U1 - Vye[ix, 2    ]
    @inbounds Vye[ix, end] = 2.0 * U2 - Vye[ix, end-1]
end

@kernel function update_τ!(τxx, τyy, τxyv, Vx, Vy, Vxe, Vye, ∇v, μs, dτ_r, dx, dy)
    ix, iy = @index(Global, NTuple)
    if @isin(τxx)  @inbounds εxx  = @∂_∂x(Vx) / dx - ∇v[ix, iy] / 3.0 end
    if @isin(τyy)  @inbounds εyy  = @∂_∂y(Vy) / dy - ∇v[ix, iy] / 3.0 end
    if @isin(τxyv) @inbounds εxyv = 0.5 * ((Vxe[ix, iy+1] - Vxe[ix, iy]) / dy + (Vye[ix+1, iy] - Vye[ix, iy]) / dx) end
    if @isin(τxx)  @inbounds τxx[ix, iy]  += (-τxx[ix, iy]  + 2.0 * μs * εxx ) * dτ_r end
    if @isin(τyy)  @inbounds τyy[ix, iy]  += (-τyy[ix, iy]  + 2.0 * μs * εyy ) * dτ_r end
    if @isin(τxyv) @inbounds τxyv[ix, iy] += (-τxyv[ix, iy] + 2.0 * μs * εxyv) * dτ_r end
end

@kernel function update_R!(Rx, Ry, Pr, τxx, τyy, τxyv, dx, dy)
    ix, iy = @index(Global, NTuple)
    if @isin(Rx) @inbounds Rx[ix, iy] = -(Pr[ix+1, iy] - Pr[ix, iy]) /dx + (τxx[ix+1, iy] - τxx[ix, iy]) / dx + (τxyv[ix+1, iy+1] - τxyv[ix+1, iy]) / dy end
    if @isin(Ry) @inbounds Ry[ix, iy] = -(Pr[ix, iy+1] - Pr[ix, iy]) /dy + (τyy[ix, iy+1] - τyy[ix, iy]) / dy + (τxyv[ix+1, iy+1] - τxyv[ix, iy+1]) / dx end
end

@kernel function update_V!(Vx, Vy, Rx, Ry, nudτ, μs)
    ix, iy = @index(Global, NTuple)
    if @isin(Vx) @inbounds Vx[ix, iy] += Rx[ix, iy] * nudτ ./ μs end
    if @isin(Vy) @inbounds Vy[ix, iy] += Ry[ix, iy] * nudτ ./ μs end
end

@views function stokes(backend=CPU(); dtype=Float64)
    # Physics
    h       = 1.0
    lx = ly = h
    μs      = 1.0
    U       = 1.0
    ρ       = 100.0 # Re = ρ * U * ly / μs
    # Numerics
    nx = ny = 100
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
    # Initialisation
    Pr      = KA.zeros(backend, dtype, nx    , ny    )
    ∇v      = KA.zeros(backend, dtype, nx    , ny    )
    Vx      = KA.zeros(backend, dtype, nx + 1, ny    )
    Vy      = KA.zeros(backend, dtype, nx    , ny + 1)
    Vxe     = KA.zeros(backend, dtype, nx + 1, ny + 2)
    Vye     = KA.zeros(backend, dtype, nx + 2, ny + 1)
    # εxx     = KA.zeros(backend, dtype, nx    , ny    )
    # εyy     = KA.zeros(backend, dtype, nx    , ny    )
    # εxyv    = KA.zeros(backend, dtype, nx + 1, ny + 1)
    τxx     = KA.zeros(backend, dtype, nx    , ny    )
    τyy     = KA.zeros(backend, dtype, nx    , ny    )
    τxyv    = KA.zeros(backend, dtype, nx + 1, ny + 1)
    Rx      = KA.zeros(backend, dtype, nx - 1, ny    )
    Ry      = KA.zeros(backend, dtype, nx    , ny - 1)
    dVx     = KA.zeros(backend, dtype, nx - 1, ny - 2)
    dVy     = KA.zeros(backend, dtype, nx - 2, ny - 1)
    Q       = KA.zeros(backend, dtype, nx + 1, ny + 1)
    UV      = KA.zeros(backend, dtype, nx + 1, ny + 1)
    RQ      = KA.zeros(backend, dtype, nx - 1, ny - 1) 
    # Action
    iter = 0; res = ϵ * 2; resv = zeros(4)
    while (max(res) > ϵ) && (iter < niter)
        update_Pr!(backend, 256, (nx+2, ny+2))(∇v, Pr, inn_y(Vxe), inn_x(Vye), Vx, Vy, r, μs, θ_dτ, dx, dy)
        bc_Vx!(backend, 256, (nx+1))(Vxe, 0.0, U)
        bc_Vy!(backend, 256, (nx+1))(Vye, 0.0, 0.0)
        update_τ!(backend, 256, (nx+2, ny+2))(τxx, τyy, τxyv, Vx, Vy, Vxe, Vye, ∇v, μs, dτ_r, dx, dy)
        update_R!(backend, 256, (nx+1, ny+1))(Rx, Ry, Pr, τxx, τyy, τxyv, dx, dy)
        update_V!(backend, 256, (nx, ny))(inn_x(Vx), inn_y(Vy), Rx, Ry, nudτ, μs)
        # Pressure
        # ∇v    .= diff(Vx, dims=1) ./ dx + diff(Vy, dims=2) ./ dy
        # Pr   .-= ∇v ./ (1.0 / (r * μs / θ_dτ))
        # Velocity + viscous rheology
        # Vxe   .= hcat(.-Vx[:,1] , Vx, 2.0 .* U .- Vx[:,end] )
        # Vye   .= vcat(.-Vy[1,:]', Vy,          .- Vy[end,:]')
        # εxx   .= diff(Vx, dims=1)./dx .- ∇v./3.0
        # εyy   .= diff(Vy, dims=2)./dy .- ∇v./3.0
        # εxyv  .= 0.5 .* (diff(Vxe, dims=2) ./ dy .+ diff(Vye, dims=1) ./ dx)
        # τxx  .+= (.-τxx  .+ 2.0 .* μs .* εxx ) .* dτ_r
        # τyy  .+= (.-τyy  .+ 2.0 .* μs .* εyy ) .* dτ_r
        # τxyv .+= (.-τxyv .+ 2.0 .* μs .* εxyv) .* dτ_r
        # Rx    .= diff(.-Pr .+ τxx, dims=1) ./ dx .+ diff(τxyv[2:end-1, :], dims=2) ./ dy
        # Ry    .= diff(.-Pr .+ τyy, dims=2) ./ dy .+ diff(τxyv[:, 2:end-1], dims=1) ./ dx
        # Advection
        # dVx   .= max.(0.0, Vx[2:end-1,2:end-1]) .* diff(Vx[1:end-1,2:end-1], dims=1) ./ dx
        # dVx  .+= min.(0.0, Vx[2:end-1,2:end-1]) .* diff(Vx[2:end  ,2:end-1], dims=1) ./ dx
        # dVx  .+= max.(0.0, avx(Vy[:,2:end-2]))  .* diff(Vx[2:end-1,1:end-1], dims=2) ./ dy
        # dVx  .+= min.(0.0, avx(Vy[:,3:end-1]))  .* diff(Vx[2:end-1,2:end  ], dims=2) ./ dy
        # dVy   .= max.(0.0, Vy[2:end-1,2:end-1]) .* diff(Vy[2:end-1,1:end-1], dims=2) ./ dy
        # dVy  .+= min.(0.0, Vy[2:end-1,2:end-1]) .* diff(Vy[2:end-1,2:end  ], dims=2) ./ dy
        # dVy  .+= max.(0.0, avy(Vx[2:end-2,:]))  .* diff(Vy[1:end-1,2:end-1], dims=1) ./ dx
        # dVy  .+= min.(0.0, avy(Vx[3:end-1,:]))  .* diff(Vy[2:end  ,2:end-1], dims=1) ./ dx
        # Rx[:,2:end-1] .-= ρ .* dVx
        # Ry[2:end-1,:] .-= ρ .* dVy
        # Update
        # Vx[2:end-1, :] .+= Rx .* nudτ ./ μs
        # Vy[:, 2:end-1] .+= Ry .* nudτ ./ μs
        # Stream function
        UV .= diff(Vxe, dims=2) ./ dy .- diff(Vye, dims=1) ./ dx
        RQ .= .-(diff(diff(Q[2:end-1, :], dims=2), dims=2) ./ dy^2 .+
                 diff(diff(Q[:, 2:end-1], dims=1), dims=1) ./ dx^2) .+ UV[2:end-1, 2:end-1] .+ (1 - 2 / nx) * RQ
        Q[2:end-1, 2:end-1] .-= dτ_Q * RQ
        (iter % ndt == 0) && (resv .= maximum.([Rx, Ry, ∇v, RQ]); res = maximum(resv); println(" iter $iter err: $(round.(resv, sigdigits=3))"))
        iter += 1
    end
    KA.synchronize(backend)
    # visualise
    fig, ax, hm = contourf(xc, yc, Array(avy(Vy)); levels=20, figure=(resolution=(1000, 800), fontsize=30), axis=(aspect=DataAspect(), title="Velocity"), colormap=:jet)
    contour!(ax, xv[2:end-1], yv[2:end-1], log10.(abs.(Array(Q[2:end-1,2:end-1]))); levels=18, color=:black)
    Colorbar(fig[:, end+1], hm); limits!(ax, -lx / 2, lx / 2, 0, ly)
    # display(fig)
    save("./out_fig.png", fig)
    return
end

stokes(ROCBackend(); dtype=Float64)

# CUDABackend()
# ROCBackend()

# fig, ax, hm = contourf(xv, yv, Q; levels=20, figure=(resolution=(1000, 800), fontsize=30), axis=(aspect=DataAspect(), title="Stream function"), colormap=:jet)