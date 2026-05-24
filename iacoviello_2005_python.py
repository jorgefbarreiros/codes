# -*- coding: utf-8 -*-
"""
Iacoviello (2005) in Python

This script:
    1. Sets the calibration.
    2. Computes the analytical steady state used in the Matlab launcher.
    3. Writes the nonlinear model equations as residuals.
    4. Numerically linearizes the model around the steady state.
    5. Solves the first-order rational expectations system using QZ.
    6. Computes IRFs to the monetary policy shock used in the Dynare file.
    7. Plots the main IRFs in one orange multipanel figure.

Requirements:
    pip install numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import ordqz


# =============================================================================
# 0. User options
# =============================================================================

HORIZON = 40

# Paper / Dynare shock:
# The .mod file has:
#     shocks;
#     var eR = 1;
#     end;
SHOCK_SIZE = 1.0

# Plot options
SAVE_FIGURE = True
FIGURE_NAME = "iacoviello_2005_irfs_orange.png"

# Set to 100.0 if you want to read log responses approximately as percent.
PLOT_SCALE = 1.0

ORANGE = "#E67E22"


# =============================================================================
# 1. Variable ordering
# =============================================================================

VARS = [
    "cp",      # patient households' consumption
    "rr",      # ex-ante real interest rate
    "b",       # debt
    "q",       # real house price
    "h",       # entrepreneurs' housing
    "infl",    # gross inflation
    "X",       # markup
    "R",       # gross nominal interest rate
    "Y",       # output
    "c",       # entrepreneurs' consumption
    "hp",      # patient households' housing
    "sdf",     # stochastic discount factor
    "lam",     # borrowing constraint multiplier
    "L",       # labor
    "vp",      # price dispersion
    "w",       # real wage
    "inflp",   # reset-price inflation
    "z1",      # Calvo auxiliary variable
    "z2",      # Calvo auxiliary variable
    "lR",      # log nominal interest rate
    "lY",      # log output
    "lpi",     # log inflation
    "lq",      # log house price
    "llam",    # log multiplier
]

IDX = {name: i for i, name in enumerate(VARS)}

PLOT_VARS = ["lR", "lY", "lpi", "lq", "llam"]

PANEL_TITLES = {
    "lR": "Nominal interest rate",
    "lY": "Output",
    "lpi": "Inflation",
    "lq": "House price",
    "llam": "Borrowing constraint multiplier",
}


def get(x, name):
    """Extract a variable from a vector using its name."""
    return float(x[IDX[name]])


def set_value(x, name, value):
    """Set a variable in a vector using its name."""
    x[IDX[name]] = value


# =============================================================================
# 2. Calibration and steady state
# =============================================================================

class Params:
    """Model calibration and steady-state objects."""

    def __init__(self):
        # Preference, technology and financial parameters
        self.beta = 0.99
        self.gamma = 0.98
        self.nu = 0.03
        self.j = 0.1
        self.m = 0.89
        self.eta = 1.01
        self.Xs = 1.1
        self.theta = 0.75

        # Taylor rule and monetary policy shock scale
        self.rY = 0.0
        self.rpi = 0.27
        self.rR = 0.73
        self.sR = 0.29

        # Objects computed exactly as in the Matlab launcher
        self.gammae = self.m * self.beta + (1.0 - self.m) * self.gamma
        self.kappa = (1.0 - self.theta) * (1.0 - self.theta * self.beta) / self.theta

        self.qhY = (self.gamma * self.nu / (1.0 - self.gammae)) * (1.0 / self.Xs)
        self.bY = (
            self.beta
            * self.m
            * self.gamma
            * self.nu
            / (1.0 - self.gammae)
            * (1.0 / self.Xs)
        )
        self.cY = (
            self.nu
            / self.Xs
            * ((1.0 - self.m * self.beta) * (1.0 - self.gamma) / (1.0 - self.gammae))
        )
        self.cpY = 1.0 - self.cY

        self.hH = (1.0 + (self.j / (1.0 - self.beta)) * self.cpY / self.qhY) ** (-1.0)
        self.hpH = 1.0 - self.hH
        self.hhp = self.hH / self.hpH
        self.iota = (1.0 - self.beta) * self.hhp

        self.epsi = self.Xs / (self.Xs - 1.0)
        self.rrb = 1.0 / self.beta

        # Normalizations
        self.Ys = 1.0
        self.Hs = 1.0

        # Steady-state levels
        self.bs = self.bY * self.Ys
        self.cs = self.cY * self.Ys
        self.cps = self.cpY * self.Ys
        self.hs = self.hH * self.Hs
        self.hps = self.hpH * self.Hs
        self.Ls = (((1.0 - self.nu) / self.Xs) / self.cpY) ** (1.0 / self.eta)
        self.As = 1.0 / (self.hs ** self.nu * self.Ls ** (1.0 - self.nu))
        self.ws = ((1.0 - self.nu) / self.Xs) * (self.Ys / self.Ls)
        self.lams = self.beta / self.cs - self.gamma / self.cs
        self.qs = self.qhY / (self.hs * self.Ys)


def steady_state(p):
    """Analytical steady state used in the original Matlab/Dynare files."""

    x = np.zeros(len(VARS))

    set_value(x, "cp", p.cps)
    set_value(x, "rr", p.rrb)
    set_value(x, "b", p.bs)
    set_value(x, "q", p.qs)
    set_value(x, "h", p.hs)
    set_value(x, "infl", 1.0)
    set_value(x, "X", p.Xs)
    set_value(x, "R", p.rrb)
    set_value(x, "Y", p.Ys)
    set_value(x, "c", p.cs)
    set_value(x, "hp", p.hps)
    set_value(x, "sdf", p.beta)
    set_value(x, "lam", p.lams)
    set_value(x, "L", p.Ls)
    set_value(x, "vp", 1.0)
    set_value(x, "w", p.ws)
    set_value(x, "inflp", 1.0)
    set_value(x, "z1", (p.Ys / p.Xs) / (1.0 - p.beta * p.theta))
    set_value(x, "z2", p.Ys / (1.0 - p.beta * p.theta))

    # Reporting variables
    set_value(x, "lR", np.log(p.rrb))
    set_value(x, "lY", np.log(p.Ys))
    set_value(x, "lpi", 0.0)
    set_value(x, "lq", np.log(p.qs))
    set_value(x, "llam", np.log(p.lams))

    return x


# =============================================================================
# 3. Nonlinear equilibrium conditions
# =============================================================================

def residual(x_lag, x_now, x_lead, eR, p):
    """
    Nonlinear residuals:
        F(x_{t-1}, x_t, E_t x_{t+1}, e_t) = 0.

    The equations follow the order of the Dynare model block.
    """

    cp = get(x_now, "cp")
    b = get(x_now, "b")
    q = get(x_now, "q")
    h = get(x_now, "h")
    infl = get(x_now, "infl")
    X = get(x_now, "X")
    R = get(x_now, "R")
    Y = get(x_now, "Y")
    c = get(x_now, "c")
    hp = get(x_now, "hp")
    sdf = get(x_now, "sdf")
    lam = get(x_now, "lam")
    L = get(x_now, "L")
    vp = get(x_now, "vp")
    w = get(x_now, "w")
    inflp = get(x_now, "inflp")
    z1 = get(x_now, "z1")
    z2 = get(x_now, "z2")
    rr = get(x_now, "rr")
    lR = get(x_now, "lR")
    lY = get(x_now, "lY")
    lpi = get(x_now, "lpi")
    lq = get(x_now, "lq")
    llam = get(x_now, "llam")

    cp_f = get(x_lead, "cp")
    c_f = get(x_lead, "c")
    q_f = get(x_lead, "q")
    L_f = get(x_lead, "L")
    X_f = get(x_lead, "X")
    infl_f = get(x_lead, "infl")
    sdf_f = get(x_lead, "sdf")
    z1_f = get(x_lead, "z1")
    z2_f = get(x_lead, "z2")

    cp_l = get(x_lag, "cp")
    b_l = get(x_lag, "b")
    h_l = get(x_lag, "h")
    infl_l = get(x_lag, "infl")
    R_l = get(x_lag, "R")
    Y_l = get(x_lag, "Y")
    vp_l = get(x_lag, "vp")

    res = np.empty(len(VARS))

    # 1. Patient households' housing Euler equation
    res[0] = q / cp - p.j / hp - p.beta * q_f / cp_f

    # 2. Patient households' labor supply
    res[1] = L ** (p.eta - 1.0) - w / cp

    # 3. Patient households' bond Euler equation
    res[2] = 1.0 / cp - p.beta * (1.0 / cp_f) * R / infl_f

    # 4. Labor demand
    res[3] = (1.0 - p.nu) * p.As * h_l ** p.nu * L ** (-p.nu) - X * w

    # 5. Entrepreneurs' housing Euler equation
    res[4] = (
        q / c
        - (p.gamma / c_f)
        * (p.nu * p.As * h ** (p.nu - 1.0) * L_f ** (1.0 - p.nu) / X_f + q_f)
        - p.m * lam * q_f * infl_f
    )

    # 6. Entrepreneurs' bond Euler equation
    res[5] = 1.0 / c - p.gamma * (1.0 / c_f) * R / infl_f - lam * R

    # 7. Borrowing constraint
    res[6] = b - p.m * (q_f * h * infl_f / R)

    # 8. Calvo auxiliary variable z1
    res[7] = z1 - Y / X - p.theta * sdf_f * infl_f ** p.epsi * z1_f

    # 9. Calvo auxiliary variable z2
    res[8] = z2 - Y - p.theta * sdf_f * infl_f ** (p.epsi - 1.0) * z2_f

    # 10. Reset-price inflation
    res[9] = inflp - (p.epsi / (p.epsi - 1.0)) * z1 / z2

    # 11. Taylor rule
    res[10] = R - (
        p.rrb ** (1.0 - p.rR)
        * R_l ** p.rR
        * (infl_l ** (1.0 + p.rpi) * (Y_l / p.Ys) ** p.rY) ** (1.0 - p.rR)
        * np.exp(p.sR * eR)
    )

    # 12. Price evolution
    res[11] = 1.0 - p.theta * infl ** (p.epsi - 1.0) - (1.0 - p.theta) * inflp ** (1.0 - p.epsi)

    # 13. Production function
    res[12] = Y * vp - p.As * h_l ** p.nu * L ** (1.0 - p.nu)

    # 14. Price dispersion
    res[13] = vp - (1.0 - p.theta) * inflp ** (-p.epsi) - p.theta * infl ** p.epsi * vp_l

    # 15. Resource constraint
    res[14] = c + cp - Y

    # 16. Housing market clearing
    res[15] = h + hp - p.Hs

    # 17. Entrepreneurs' budget constraint
    res[16] = b - c - q * (h - h_l) - R_l * b_l / infl - w * L + Y * vp / X

    # 18. Stochastic discount factor
    res[17] = sdf - p.beta * cp_l / cp

    # 19. Ex-ante real interest rate
    res[18] = rr - R / infl_f

    # 20-24. Reporting variables
    res[19] = lR - np.log(R)
    res[20] = lY - np.log(Y)
    res[21] = lpi - np.log(infl)
    res[22] = lq - np.log(q)
    res[23] = llam - np.log(lam)

    return res


# =============================================================================
# 4. Numerical linearization
# =============================================================================

def numerical_jacobian(p, x_ss, step_scale=1e-6):
    """
    Central-difference numerical derivatives.

    Linearized system:
        A x_{t+1} + B x_t + C x_{t-1} + D e_t = 0

    where x denotes deviations from steady state.
    """

    n = len(VARS)

    A = np.zeros((n, n))
    B = np.zeros((n, n))
    C = np.zeros((n, n))
    D = np.zeros((n, 1))

    for j in range(n):
        step = step_scale * max(abs(x_ss[j]), 1.0)
        dx = np.zeros(n)
        dx[j] = step

        A[:, j] = (
            residual(x_ss, x_ss, x_ss + dx, 0.0, p)
            - residual(x_ss, x_ss, x_ss - dx, 0.0, p)
        ) / (2.0 * step)

        B[:, j] = (
            residual(x_ss, x_ss + dx, x_ss, 0.0, p)
            - residual(x_ss, x_ss - dx, x_ss, 0.0, p)
        ) / (2.0 * step)

        C[:, j] = (
            residual(x_ss + dx, x_ss, x_ss, 0.0, p)
            - residual(x_ss - dx, x_ss, x_ss, 0.0, p)
        ) / (2.0 * step)

    e_step = step_scale
    D[:, 0] = (
        residual(x_ss, x_ss, x_ss, e_step, p)
        - residual(x_ss, x_ss, x_ss, -e_step, p)
    ) / (2.0 * e_step)

    return A, B, C, D


# =============================================================================
# 5. First-order solution using QZ
# =============================================================================

def stable_selector(alpha, beta, stake):
    """Select generalized eigenvalues alpha / beta inside the unit circle."""

    eig_abs = np.full_like(np.real(alpha), fill_value=np.inf, dtype=float)
    ok = np.abs(beta) > 1e-12
    eig_abs[ok] = np.abs(alpha[ok] / beta[ok])

    return eig_abs < stake


def solve_first_order_qz(A, B, C, D, x_ss, stake=1.0 - 1e-8):
    """
    Solve:
        A E_t[x_{t+1}] + B x_t + C x_{t-1} + D e_t = 0

    Decision rule:
        x_t = P x_{t-1} + Q e_t
    """

    n = A.shape[0]

    zeros = np.zeros((n, n))
    eye = np.eye(n)

    G0 = np.block([
        [A, zeros],
        [zeros, eye]
    ])

    G1 = np.block([
        [-B, -C],
        [eye, zeros]
    ])

    def sort_fun(alpha, beta):
        return stable_selector(alpha, beta, stake)

    # Generalized Schur decomposition
    _, _, alpha, beta, _, Z = ordqz(G1, G0, sort=sort_fun, output="complex")

    eig = np.full(alpha.shape, np.inf + 0j, dtype=complex)
    ok = np.abs(beta) > 1e-12
    eig[ok] = alpha[ok] / beta[ok]

    n_stable = int(np.sum(np.abs(eig) < stake))

    if n_stable != n:
        raise RuntimeError(
            "QZ did not find exactly n stable roots. "
            f"Required: {n}; found: {n_stable}."
        )

    Z_stable = Z[:, :n]
    Z_top = Z_stable[:n, :]
    Z_bottom = Z_stable[n:, :]

    if np.linalg.cond(Z_bottom) > 1e12:
        raise RuntimeError("The stable subspace is close to singular.")

    P = Z_top @ np.linalg.inv(Z_bottom)
    P = np.real_if_close(P, tol=1000).real

    Q = -np.linalg.solve(A @ P + B, D)
    Q = np.real_if_close(Q, tol=1000).real

    return P, Q, eig, n_stable


def solve_model(p):
    """Compute steady state, linearize and solve the model."""

    x_ss = steady_state(p)

    ss_resid = residual(x_ss, x_ss, x_ss, 0.0, p)
    max_ss_resid = np.max(np.abs(ss_resid))

    if max_ss_resid > 1e-7:
        raise RuntimeError(f"Steady-state residuals are too large: {max_ss_resid:.3e}")

    A, B, C, D = numerical_jacobian(p, x_ss)

    P, Q, eig, n_stable = solve_first_order_qz(A, B, C, D, x_ss)

    return {
        "x_ss": x_ss,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "P": P,
        "Q": Q,
        "eig": eig,
        "n_stable": n_stable,
    }


# =============================================================================
# 6. Impulse response functions
# =============================================================================

def impulse_response(solution, horizon=40, shock_size=1.0):
    """
    Compute IRFs in deviations from steady state.

    Decision rule:
        x_t = P x_{t-1} + Q e_t
    """

    P = solution["P"]
    Q = solution["Q"]

    n = len(VARS)
    irf = np.zeros((horizon, n))

    irf[0, :] = Q[:, 0] * shock_size

    for t in range(1, horizon):
        irf[t, :] = P @ irf[t - 1, :]

    return irf


# =============================================================================
# 7. Plotting
# =============================================================================

def plot_irfs(irf, variables, scale=1.0):
    """Plot selected IRFs in one orange multipanel figure."""

    periods = np.arange(irf.shape[0])

    n_vars = len(variables)
    ncols = 3
    nrows = int(np.ceil(n_vars / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(14, 7),
        squeeze=False
    )

    axes_flat = axes.ravel()

    for ax, var in zip(axes_flat, variables):
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)

        ax.plot(
            periods,
            scale * irf[:, IDX[var]],
            color=ORANGE,
            linewidth=2.5
        )

        ax.set_title(PANEL_TITLES[var], fontsize=12)
        ax.set_xlabel("Periods")
        ax.set_ylabel("Response")
        ax.grid(True, alpha=0.25)

    # Hide unused panels
    for ax in axes_flat[n_vars:]:
        ax.axis("off")

    fig.suptitle(
        "Iacoviello (2005): IRFs to a Monetary Policy Shock",
        fontsize=15,
        y=0.98
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    if SAVE_FIGURE:
        fig.savefig(FIGURE_NAME, dpi=300, bbox_inches="tight")

    plt.show()


# =============================================================================
# 8. Run
# =============================================================================

p = Params()
solution = solve_model(p)

irf = impulse_response(
    solution,
    horizon=HORIZON,
    shock_size=SHOCK_SIZE
)

# Diagnostics
x_ss = solution["x_ss"]
A = solution["A"]
B = solution["B"]
C = solution["C"]
P = solution["P"]
eig = solution["eig"]

ss_resid = residual(x_ss, x_ss, x_ss, 0.0, p)
quad_err = A @ P @ P + B @ P + C

print("Iacoviello (2005) Python solution")
print("---------------------------------")
print(f"Stable roots selected: {solution['n_stable']} out of {len(VARS)} required")
print(f"Maximum steady-state residual: {np.max(np.abs(ss_resid)):.3e}")
print(f"Maximum |A P^2 + B P + C|: {np.max(np.abs(quad_err)):.3e}")
print("")
print("Impact responses, period 0")
print("--------------------------")
for var in PLOT_VARS:
    print(f"{PANEL_TITLES[var]:35s}: {irf[0, IDX[var]]: .10f}")

plot_irfs(
    irf=irf,
    variables=PLOT_VARS,
    scale=PLOT_SCALE
)
