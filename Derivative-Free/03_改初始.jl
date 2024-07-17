using Random
using Distributions
using LinearAlgebra
using ForwardDiff
include("../Inversion/Plot.jl")
include("../Inversion/GMGD.jl")




##########
function Gaussian_mixture(x, args)
    x_w, x_mean, inv_sqrt_x_cov = args
    # C = L L.T
    # C^-1 = L^-TL^-1
    N_x = size(x_mean, 2)
    ρ = 0
    for im = 1:length(x_w)
        ρ += x_w[im]*exp(-0.5*(x-x_mean[im,:])'*(inv_sqrt_x_cov[im]'*inv_sqrt_x_cov[im]*(x-x_mean[im,:])))/det(inv_sqrt_x_cov[im])
    end
    return log(ρ) - N_x/2*log(2*π)
end



function Gaussian_mixture_V(θ, args)
    return -Gaussian_mixture(θ, args), 
           -ForwardDiff.gradient(x -> Gaussian_mixture(x, args), θ), 
           -ForwardDiff.hessian(x -> Gaussian_mixture(x, args), θ)
end
##########


function G(θ)
    #return θ
    return [θ'*θ; θ]
end


function F(θ, args)
    y, ση = args
    Gθ = G(θ)
    return (y - Gθ) ./ ση
end


function logrho(θ, args)
    Fθ = F(θ, args)
    return -0.5*norm(Fθ)^2
end


function V(θ, args)
    return -logrho(θ, args), 
           -ForwardDiff.gradient(x -> logrho(x, args), θ), 
           -ForwardDiff.hessian(x -> logrho(x, args), θ)
end


##########
function Gaussian_mixture_VI(func_V, func_F, w0, μ0, Σ0; N_iter = 100, dt = 1.0e-3)

    N_modes, N_θ = size(μ0)
    

    
    T =  N_iter * dt
    N_modes = 1
    x0_w = w0
    x0_mean = μ0
    xx0_cov = Σ0
    sqrt_matrix_type = "Cholesky"
    quadrature_type_GM = "cubature_transform_o5"
    
    objs = []

    if func_V !== nothing
#         gmgdobj = GMGD_Run(
#         func_V, 
#         T,
#         N_iter,
#         # Initial condition
#         x0_w, x0_mean, xx0_cov;
#         sqrt_matrix_type = sqrt_matrix_type,
#         # setup for Gaussian mixture part
#         quadrature_type_GM = quadrature_type_GM,
#         # setup for potential function part
#         Bayesian_inverse_problem = false, 
#         quadrature_type = "cubature_transform_o3")
        
        
        gmgdobj = GMGD_Run(
        func_V, 
        T,
        N_iter,
        # Initial condition
        x0_w, x0_mean, xx0_cov;
        sqrt_matrix_type = sqrt_matrix_type,
        # setup for Gaussian mixture part
        quadrature_type_GM = "mean_point",
        # setup for potential function part
        Bayesian_inverse_problem = false, 
        quadrature_type = "mean_point")
        
        push!(objs, gmgdobj)

    end

    if func_F !== nothing
        N_f = length(func_F(ones(N_θ)))
        gmgdobj_BIP = GMGD_Run(
        func_F, 
        T,
        N_iter,
        # Initial condition
        x0_w, x0_mean, xx0_cov;
        sqrt_matrix_type = sqrt_matrix_type,
        # setup for Gaussian mixture part
        quadrature_type_GM = "mean_point",
        # setup for potential function part
        Bayesian_inverse_problem = true, 
        N_f = N_f,
        quadrature_type = "unscented_transform",
        c_weight_BIP = 1e-3,
        w_min=1e-10)
        
        push!(objs, gmgdobj_BIP)

    end

    return objs
end


##########

include("../Inversion/GMGD.jl")
fig, ax = PyPlot.subplots(nrows=4, ncols=3, sharex=false, sharey=false, figsize=(25,20))


N_modes = 20
lines = 10



N_iter = 1000
Nx, Ny = 100,100
ση = [0.2; 2.0]
Gtype = "1D"
y = [1.0; 3.0]
func_args = (y, ση, 0 , Gtype)
func_F(x) = F(x, func_args)
func_dV(x) = V(x, func_args)
objs = []
half_objs = []
for i = 1 : lines
    Random.seed!(10 + i);
    x0_w  = ones(N_modes)/N_modes
    μ0, Σ0 = [3.], [4.]
    N_x = length(μ0)
    x0_mean, xx0_cov = zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)
    for im = 1:N_modes
        x0_mean[im, :]    .= rand(MvNormal(zeros(N_x), Σ0)) + μ0
        xx0_cov[im, :, :] .= Σ0
    end
    push!(half_objs, Gaussian_mixture_VI(nothing, func_F, x0_w[1:div(N_modes,2)], x0_mean[1:div(N_modes,2),:], xx0_cov[1:div(N_modes,2),:,:]; N_iter = N_iter, dt = 1e-2)[1])
    push!(objs, Gaussian_mixture_VI(nothing, func_F, x0_w, x0_mean, xx0_cov; N_iter = N_iter, dt = 1e-2)[1])
end
visualization_1d_multi(ax[1,:]; Nx = Nx, x_lim=[-4.0, 4.0], func_F=func_F, objs=objs, half_objs=half_objs, lines = lines)



ση = [0.5; 2.0]
Gtype = "1D"
func_args = (y, ση, 0, Gtype)
func_F(x) = F(x, func_args)
func_dV(x) = V(x, func_args)
objs = []
half_objs = []
for i = 1 : lines
    Random.seed!(10 + i);
    x0_w  = ones(N_modes)/N_modes
    μ0, Σ0 = [3.], [4.]
    N_x = length(μ0)
    x0_mean, xx0_cov = zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)
    for im = 1:N_modes
        x0_mean[im, :]    .= rand(MvNormal(zeros(N_x), Σ0)) + μ0
        xx0_cov[im, :, :] .= Σ0
    end
    push!(half_objs, Gaussian_mixture_VI(nothing, func_F, x0_w[1:div(N_modes,2)], x0_mean[1:div(N_modes,2),:], xx0_cov[1:div(N_modes,2),:,:]; N_iter = N_iter, dt = 1e-2)[1])
    push!(objs, Gaussian_mixture_VI(nothing, func_F, x0_w, x0_mean, xx0_cov; N_iter = N_iter, dt = 1e-2)[1])
end
visualization_1d_multi(ax[2,:]; Nx = Nx, x_lim=[-4.0, 4.0], func_F=func_F, objs=objs, half_objs=half_objs, lines = lines)



ση = [1.0; 2.0]
Gtype = "1D"
func_args = (y, ση, 0 , Gtype)
func_F(x) = F(x, func_args)
func_dV(x) = V(x, func_args)
objs = []
half_objs = []
for i = 1 : lines
    Random.seed!(10 + i);
    x0_w  = ones(N_modes)/N_modes
    μ0, Σ0 = [3.], [4.]
    N_x = length(μ0)
    x0_mean, xx0_cov = zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)
    for im = 1:N_modes
        x0_mean[im, :]    .= rand(MvNormal(zeros(N_x), Σ0)) + μ0
        xx0_cov[im, :, :] .= Σ0
    end
    push!(half_objs, Gaussian_mixture_VI(nothing, func_F, x0_w[1:div(N_modes,2)], x0_mean[1:div(N_modes,2),:], xx0_cov[1:div(N_modes,2),:,:]; N_iter = N_iter, dt = 1e-2)[1])
    push!(objs, Gaussian_mixture_VI(nothing, func_F, x0_w, x0_mean, xx0_cov; N_iter = N_iter, dt = 1e-2)[1])
end
visualization_1d_multi(ax[3,:]; Nx = Nx, x_lim=[-4.0, 4.0], func_F=func_F, objs=objs, half_objs=half_objs, lines = lines)



ση = [1.5; 2.0]
Gtype = "1D"
func_args = (y, ση, 0 , Gtype)
func_F(x) = F(x, func_args)
func_dV(x) = V(x, func_args)
objs = []
half_objs = []
for i = 1 : lines
    Random.seed!(10 + i);
    x0_w  = ones(N_modes)/N_modes
    μ0, Σ0 = [3.], [4.]
    N_x = length(μ0)
    x0_mean, xx0_cov = zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)
    for im = 1:N_modes
        x0_mean[im, :]    .= rand(MvNormal(zeros(N_x), Σ0)) + μ0
        xx0_cov[im, :, :] .= Σ0
    end
    push!(half_objs, Gaussian_mixture_VI(nothing, func_F, x0_w[1:div(N_modes,2)], x0_mean[1:div(N_modes,2),:], xx0_cov[1:div(N_modes,2),:,:]; N_iter = N_iter, dt = 1e-2)[1])
    push!(objs, Gaussian_mixture_VI(nothing, func_F, x0_w, x0_mean, xx0_cov; N_iter = N_iter, dt = 1e-2)[1])
end
visualization_1d_multi(ax[4,:]; Nx = Nx, x_lim=[-4.0, 4.0], func_F=func_F, objs=objs, half_objs=half_objs, lines = lines)



# ση = [0.3; 1.0; 1.0]
# Gtype = "Double_banana"
# λ = 100.0
# y = [log(λ+1); 0.0; 0.0]
# func_args = (y, ση, λ , Gtype)
# func_F(x) = F(x, func_args)
# func_dV(x) = V(x, func_args)
# objs = (Gaussian_mixture_VI(nothing, func_F, x0_w[1:div(N_modes,2)], x0_mean[1:div(N_modes,2),:], xx0_cov[1:div(N_modes,2),:,:]; N_iter = N_iter, dt = 1e-1)[1],
#         Gaussian_mixture_VI(nothing, func_F, x0_w, x0_mean, xx0_cov; N_iter = N_iter, dt = 1e-1)[1])
# visualization_2d(ax[5,:]; Nx = Nx, Ny = Ny, x_lim=[-3.0, 3.0], y_lim=[-3.0, 3.0], func_F=func_F, objs=objs)

fig.tight_layout()
fig.savefig("DFGMGD.pdf")

#########

