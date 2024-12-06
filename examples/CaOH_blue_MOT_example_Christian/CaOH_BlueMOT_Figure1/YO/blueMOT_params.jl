### SIMULATION PARAMETERS: 3-FREQUENCY 1+2 BLUE MOT  ###

# DEFINE STATES #
energy_offset = (2π / Γ) * energy(states[13])
energies = energy.(states) .* (2π / Γ)

# DEFINE FREQUENCIES #
detuning = +5.0
δ1 = +0.00
δ2 = +3.00
δ3 = -0.60
δ4 = +0.60

Δ1 = 1e6 * (detuning + δ1)
Δ2 = 1e6 * (detuning + δ2)
Δ3 = 1e6 * (detuning + δ3)
Δ4 = 1e6 * (detuning + δ4)

f1 = energy(states[end]) - energy(states[10]) + Δ1
f2 = energy(states[end]) - energy(states[9]) + Δ2
f3 = energy(states[end]) - energy(states[1]) + Δ3
f4 = energy(states[end]) - energy(states[1]) + Δ4

freqs = [f1, f2, f3, f4] .* (2π / Γ)

# DEFINE SATURATION INTENSITIES #
beam_radius = 5e-3
Isat = π*h*c*Γ/(3λ^3)
P = @with_unit 16 "mW"
I = 2P / (π * beam_radius^2)

total_sat = I / Isat
s1 = 0.28total_sat
s2 = 0.16total_sat
s3 = 0.28total_sat
s4 = 0.28total_sat

sats = [s1, s2, s3, s4]

# DEFINE POLARIZATIONS #
pols = [σ⁺, σ⁺, σ⁺, σ⁻]