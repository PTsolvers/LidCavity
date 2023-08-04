# LidCavity

Lid driven cavity in PT fashion for laminar flow regimes (Re < ~2000). For turbulent flow, check out [NavierStokes.jl](https://github.com/utkinis/NavierStokes.jl).

## Output

Result for Reynolds number `Re = 100`

<p align="center">
    <img src="docs/output.png" alt="LidCavity" width="500">
</p>

## Codes
- [`lid_cavity.jl`](lid_cavity.jl) plain Julia code (vectorised)
- [`lid_cavity_ka_1.jl`](lid_cavity_ka_1.jl) backend agnostic [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl) based code
