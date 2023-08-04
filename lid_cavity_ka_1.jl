using CairoMakie
using KernelAbstractions
const KA = KernelAbstractions

@views amean1(A) = 0.5 .* (A[1:end-1] .+ A[2:end])
@views avx(A) = 0.5 .* (A[1:end-1, :] .+ A[2:end, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])
inn(A) = @view A[2:end-1,2:end-1]
inn_x(A) = @view A[2:end-1,:]
inn_y(A) = @view A[:,2:end-1]

macro isin(A) esc(:( checkbounds(Bool, $A, ix, iy) )) end
macro all(A) esc(:( $A[ix, iy] )) end
macro inn(A) esc(:( $A[ix+1, iy+1] )) end
macro ∂_∂x(A) esc(:( $A[ix+1, iy] - $A[ix, iy] )) end
macro ∂_∂y(A) esc(:( $A[ix, iy+1] - $A[ix, iy] )) end
macro ∂2_∂x(A) esc(:( $A[ix+2, iy+1] - 2.0 * $A[ix+1, iy+1] + $A[ix, iy+1] )) end
macro ∂2_∂y(A) esc(:( $A[ix+1, iy+2] - 2.0 * $A[ix+1, iy+1] + $A[ix+1, iy] )) end
macro avx(A, ix, iy) esc(:( 0.5 * ($A[$ix, $iy] + $A[$ix+1, $iy]) )) end
macro avy(A, ix, iy) esc(:( 0.5 * ($A[$ix, $iy] + $A[$ix, $iy+1]) )) end

@kernel function update_Pr!(∇v, Pr, Vxe, Vye, Vx, Vy, r, μs, θ_dτ, dx, dy)
    ix, iy = @index(Global, NTuple)
    if @isin(∇v) @inbounds @all(∇v)  = @∂_∂x(Vx) / dx + @∂_∂y(Vy) / dy end
    if @isin(Pr) @inbounds @all(Pr) -= @all(∇v) / (1.0 / (r * μs / θ_dτ)) end
    if @isin(Vx) @inbounds @all(Vxe) = @all(Vx) end
    if @isin(Vy) @inbounds @all(Vye) = @all(Vy) end
end

@kernel function set_bc_Vx!(Vxe, Vx, U)
    ix = @index(Global, Linear)
    @inbounds Vxe[ix, 1  ] =         - Vx[ix, 1  ]
    @inbounds Vxe[ix, end] = 2.0 * U - Vx[ix, end]
end

@kernel function set_bc_Vy!(Vye, Vy)
    iy = @index(Global, Linear)
    @inbounds Vye[1  , iy] = - Vy[1  , iy]
    @inbounds Vye[end, iy] = - Vy[end, iy]
end

@kernel function update_τ!(τxx, τyy, τxyv, Vx, Vy, Vxe, Vye, ∇v, μs, dτ_r, dx, dy, UV)
    ix, iy = @index(Global, NTuple)
    if @isin(τxx)  @inbounds εxx  = @∂_∂x(Vx) / dx - @all(∇v) / 3.0 end
    if @isin(τyy)  @inbounds εyy  = @∂_∂y(Vy) / dy - @all(∇v) / 3.0 end
    if @isin(τxyv) @inbounds εxyv = 0.5 * (@∂_∂y(Vxe) / dy + @∂_∂x(Vye) / dx) end
    if @isin(τxx)  @inbounds @all(τxx)  += (-@all(τxx)  + 2.0 * μs * εxx ) * dτ_r end
    if @isin(τyy)  @inbounds @all(τyy)  += (-@all(τyy)  + 2.0 * μs * εyy ) * dτ_r end
    if @isin(τxyv) @inbounds @all(τxyv) += (-@all(τxyv) + 2.0 * μs * εxyv) * dτ_r end
    if @isin(UV)   @inbounds @all(UV) = @∂_∂y(Vxe) / dy - @∂_∂x(Vye) / dx end
end

@kernel function update_R!(Rx, Ry, Pr, τxx, τyy, τxyv, dx, dy, RQ, Q, UV, nx)
    ix, iy = @index(Global, NTuple)
    if @isin(Rx) @inbounds @all(Rx) = - @∂_∂x(Pr) /dx + @∂_∂x(τxx) / dx + (τxyv[ix+1, iy+1] - τxyv[ix+1, iy]) / dy end
    if @isin(Ry) @inbounds @all(Ry) = - @∂_∂y(Pr) /dy + @∂_∂y(τyy) / dy + (τxyv[ix+1, iy+1] - τxyv[ix, iy+1]) / dx end
    if @isin(RQ) @inbounds @all(RQ) = - (@∂2_∂y(Q) / dy^2 + @∂2_∂x(Q) / dx^2) + @inn(UV) + (1 - 2 / nx) * @all(RQ) end
end

@kernel function advect_V!(dVx, dVy, Vx, Vy, dx, dy)
    ix, iy = @index(Global, NTuple)
    if @isin(dVx) @inbounds @all(dVx)  = max(0.0,      Vx[ix+1, iy+1]) * (Vx[ix+1, iy+1] - Vx[ix  , iy+1]) / dx end
    if @isin(dVx) @inbounds @all(dVx) += min(0.0,      Vx[ix+1, iy+1]) * (Vx[ix+2, iy+1] - Vx[ix+1, iy+1]) / dx end
    if @isin(dVx) @inbounds @all(dVx) += max(0.0, @avx(Vy, ix , iy+1)) * (Vx[ix+1, iy+1] - Vx[ix+1, iy  ]) / dy end
    if @isin(dVx) @inbounds @all(dVx) += min(0.0, @avx(Vy, ix , iy+2)) * (Vx[ix+1, iy+2] - Vx[ix+1, iy+1]) / dy end
    if @isin(dVy) @inbounds @all(dVy)  = max(0.0,      Vy[ix+1, iy+1]) * (Vy[ix+1, iy+1] - Vy[ix+1, iy  ]) / dy end
    if @isin(dVy) @inbounds @all(dVy) += min(0.0,      Vy[ix+1, iy+1]) * (Vy[ix+1, iy+2] - Vy[ix+1, iy+1]) / dy end
    if @isin(dVy) @inbounds @all(dVy) += max(0.0, @avy(Vx, ix+1, iy )) * (Vy[ix+1, iy+1] - Vy[ix  , iy+1]) / dx end
    if @isin(dVy) @inbounds @all(dVy) += min(0.0, @avy(Vx, ix+2, iy )) * (Vy[ix+2, iy+1] - Vy[ix+1 ,iy+1]) / dx end
end

@kernel function update_R!(Rx, Ry, dVx, dVy, ρ)
    ix, iy = @index(Global, NTuple)
    if @isin(Rx) @inbounds @all(Rx) -= ρ * @all(dVx) end
    if @isin(Ry) @inbounds @all(Ry) -= ρ * @all(dVy) end
end

@kernel function update_V!(Vx, Vy, Rx, Ry, nudτ, μs, Q, RQ, dτ_Q)
    ix, iy = @index(Global, NTuple)
    if @isin(Vx) @inbounds @all(Vx) += @all(Rx) * nudτ ./ μs end
    if @isin(Vy) @inbounds @all(Vy) += @all(Ry) * nudτ ./ μs end
    if @isin(Q)  @inbounds @all(Q)  -= @all(RQ) * dτ_Q end
end

@views function stokes(be=CPU(); dat=Float64)
    # Physics
    h       = 1.0
    lx = ly = h
    μs      = 1.0
    U       = 1.0
    ρ       = 100.0 # Re = ρ * U * ly / μs
    # Numerics
    nx = ny = 128
    ndt     = 10nx
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
    Pr      = KA.zeros(be, dat, nx    , ny    )
    ∇v      = KA.zeros(be, dat, nx    , ny    )
    Vx      = KA.zeros(be, dat, nx + 1, ny    )
    Vy      = KA.zeros(be, dat, nx    , ny + 1)
    Vxe     = KA.zeros(be, dat, nx + 1, ny + 2)
    Vye     = KA.zeros(be, dat, nx + 2, ny + 1)
    τxx     = KA.zeros(be, dat, nx    , ny    )
    τyy     = KA.zeros(be, dat, nx    , ny    )
    τxyv    = KA.zeros(be, dat, nx + 1, ny + 1)
    Rx      = KA.zeros(be, dat, nx - 1, ny    )
    Ry      = KA.zeros(be, dat, nx    , ny - 1)
    dVx     = KA.zeros(be, dat, nx - 1, ny - 2)
    dVy     = KA.zeros(be, dat, nx - 2, ny - 1)
    Q       = KA.zeros(be, dat, nx + 1, ny + 1)
    UV      = KA.zeros(be, dat, nx + 1, ny + 1)
    RQ      = KA.zeros(be, dat, nx - 1, ny - 1) 
    # Action
    iter = 0; res = ϵ * 2; resv = zeros(4)
    while (max(res) > ϵ) && (iter < niter)
        update_Pr!(be, 256, (nx+2, ny+2))(∇v, Pr, inn_y(Vxe), inn_x(Vye), Vx, Vy, r, μs, θ_dτ, dx, dy)
        set_bc_Vx!(be, 256, (nx+1))(Vxe, Vx, U)
        set_bc_Vy!(be, 256, (ny+1))(Vye, Vy)
        update_τ!(be, 256, (nx+2, ny+2))(τxx, τyy, τxyv, Vx, Vy, Vxe, Vye, ∇v, μs, dτ_r, dx, dy, UV)
        update_R!(be, 256, (nx+1, ny+1))(Rx, Ry, Pr, τxx, τyy, τxyv, dx, dy, RQ, Q, UV, nx)
        advect_V!(be, 256, (nx-1, ny-1))(dVx, dVy, Vx, Vy, dx, dy)
        update_R!(be, 256, (nx, ny))(inn_y(Rx), inn_x(Ry), dVx, dVy, ρ)
        update_V!(be, 256, (nx, ny))(inn_x(Vx), inn_y(Vy), Rx, Ry, nudτ, μs, inn(Q), RQ, dτ_Q)
        (iter % ndt == 0) && (resv .= maximum.([Rx, Ry, ∇v, RQ]); res = maximum(resv); println(" iter/nx $(round(iter/nx)) err: $(round.(resv, sigdigits=3))"))
        iter += 1
    end
    KA.synchronize(be)
    # visualise
    fig, ax, hm = contourf(xc, yc, Array(avy(Vy)); levels=20, figure=(resolution=(1000, 800), fontsize=30), axis=(aspect=DataAspect(), title="Velocity"), colormap=:jet)
    contour!(ax, xv[2:end-1], yv[2:end-1], log10.(abs.(Array(Q[2:end-1,2:end-1]))); levels=18, color=:black)
    Colorbar(fig[:, end+1], hm); limits!(ax, -lx / 2, lx / 2, 0, ly)
    # display(fig)
    save("./out_fig.png", fig)
    return
end

# CUDABackend() for running on Nvidia GPU (needs using CUDA on top)
# ROCBackend() for running on AMD GPU (needs using AMDGPU on top)
stokes(CPU(); dat=Float64)
