import numpy as numpy

# set params
β_annual = 0.96
β = β_annual ** (20)
σ = 2.2
n_vec = np.array([1., 1., 0.2])
α = 0.35
A = 1.
δ_annual = .05
δ = 1 - (1 - δ_annual) ** 20

T = 50
path_tol = 1e-9 # ε

ξ = 0.2 # has to be in (0,1) but can't be too close to 1. In practice, <0.2

# Solve for the steady state 
b_2bar, b_3bar = get_SS() # writing it a little differently ("cheating") than described in the lab

# Set initial condition 
b_21 = .8 * b_2bar
b_31 = 1.1 * b_3bar

# Initial guess for the path of K
K_1 = b_21 + b_31 # doing a linear path
K_bar = b_2bar + b_3bar
K_path = np.linspace(K_1, K_bar, T)

r_path = get_r(K_path, stuff) # fill in the stuff
w_path = get_w(K_pth, stuff) # fill in the stuff

def get_a_life(b_1, r_path, w_path, n_vec, β, σ):

    # calls an optimizer

    return b_3


def get_some_lives(b_1, r_path, w_path, n_vec, β, σ):

    # calls an optimizer

    return b_2, b_3

b_mx = np.zeros((T+1, 2))
diag_mask = np.eye(2, dtype=bool)
for t in range(T+1):
    # solve for b_2 and b_3 
    break

K_path_pr = b_mx[:-1].sum(axis=1) # K_path_prime

distance = ((K_path_pr - K_path)**2).sum() # L2 norm

# put in a while loop -- while distance > tolerance, and while the 
# iteration number is less than max iter, do this..
# once the distance between k path and k path prime, we know that 
# b mx represents optimal household decisions, can use the agg
# capital stock to get interest rates, etc etc.



diag_mask # test

# You can parameterize the inequality in a steady state as much as you want
# You can predict what the economy is going to do between now and the steady state

