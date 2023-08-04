using CairoMakie

@views amean1(A) = 0.5 .* (A[1:end-1] .+ A[2:end])
@views avx(A) = 0.5 .* (A[1:end-1, :] .+ A[2:end, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])

@views function stokes()
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
    Pr      = zeros(nx    , ny    )
    ∇v      = zeros(nx    , ny    )
    Vx      = zeros(nx + 1, ny    )
    Vy      = zeros(nx    , ny + 1)
    Vxe     = zeros(nx + 1, ny + 2)
    Vye     = zeros(nx + 2, ny + 1)
    εxx     = zeros(nx    , ny    )
    εyy     = zeros(nx    , ny    )
    εxyv    = zeros(nx + 1, ny + 1)
    τxx     = zeros(nx    , ny    )
    τyy     = zeros(nx    , ny    )
    τxyv    = zeros(nx + 1, ny + 1)
    Rx      = zeros(nx - 1, ny    )
    Ry      = zeros(nx    , ny - 1)
    dVx     = zeros(nx - 1, ny - 2)
    dVy     = zeros(nx - 2, ny - 1)
    Q       = zeros(nx + 1, ny + 1)
    UV      = zeros(nx + 1, ny + 1)
    RQ      = zeros(nx - 1, ny - 1) 
    # Action
    iter = 0; res = ϵ * 2; resv = zeros(4)
    while (max(res) > ϵ) && (iter < niter)
        # Pressure
        ∇v    .= diff(Vx, dims=1) ./ dx + diff(Vy, dims=2) ./ dy
        Pr   .-= ∇v ./ (1.0 / (r * μs / θ_dτ))
        # Velocity + viscous rheology
        Vxe   .= hcat(.-Vx[:,1] , Vx, 2.0 .* U .- Vx[:,end] )
        Vye   .= vcat(.-Vy[1,:]', Vy,          .- Vy[end,:]')
        εxx   .= diff(Vx, dims=1)./dx .- ∇v./3.0
        εyy   .= diff(Vy, dims=2)./dy .- ∇v./3.0
        εxyv  .= 0.5 .* (diff(Vxe, dims=2) ./ dy .+ diff(Vye, dims=1) ./ dx)
        τxx  .+= (.-τxx  .+ 2.0 .* μs .* εxx ) .* dτ_r
        τyy  .+= (.-τyy  .+ 2.0 .* μs .* εyy ) .* dτ_r
        τxyv .+= (.-τxyv .+ 2.0 .* μs .* εxyv) .* dτ_r
        Rx    .= diff(.-Pr .+ τxx, dims=1) ./ dx .+ diff(τxyv[2:end-1, :], dims=2) ./ dy
        Ry    .= diff(.-Pr .+ τyy, dims=2) ./ dy .+ diff(τxyv[:, 2:end-1], dims=1) ./ dx
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
        Vx[2:end-1, :] .+= Rx .* nudτ ./ μs
        Vy[:, 2:end-1] .+= Ry .* nudτ ./ μs
        # Stream function
        UV .= diff(Vxe, dims=2) ./ dy .- diff(Vye, dims=1) ./ dx
        RQ .= .-(diff(diff(Q[2:end-1, :], dims=2), dims=2) ./ dy^2 .+
                 diff(diff(Q[:, 2:end-1], dims=1), dims=1) ./ dx^2) .+ UV[2:end-1, 2:end-1] .+ (1 - 2 / nx) * RQ
        Q[2:end-1, 2:end-1] .-= dτ_Q * RQ
        (iter % ndt == 0) && (resv .= maximum.([Rx, Ry, ∇v, RQ]); res = maximum(resv); println(" iter $iter err: $(round.(resv, sigdigits=3))"))
        iter += 1
    end
    # visualise
    fig, ax, hm = contourf(xc, yc, avy(Vy); levels=20, figure=(resolution=(1000, 800), fontsize=30), axis=(aspect=DataAspect(), title="Velocity"), colormap=:jet)
    contour!(ax, xv[2:end-1], yv[2:end-1], log10.(abs.(Q[2:end-1,2:end-1])); levels=18, color=:black)
    Colorbar(fig[:, end+1], hm); limits!(ax, -lx / 2, lx / 2, 0, ly)
    # display(fig)
    save("./out_fig.png", fig)
    return
end

stokes()

# fig, ax, hm = contourf(xv, yv, Q; levels=20, figure=(resolution=(1000, 800), fontsize=30), axis=(aspect=DataAspect(), title="Stream function"), colormap=:jet)