#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python translation of the nonlinear Dynare implementation of Iacoviello (2005).

Original Dynare/Matlab workflow represented in the PDFs supplied by the user:
    1. execute_iacoviello_nonlin.m
       - calibrates parameters,
       - computes the steady state,
       - calls Dynare,
       - stores selected IRFs.
    2. iacoviello_2005_base_nonlin.mod
       - contains the nonlinear equilibrium equations,
       - initializes the steady state,
       - runs stoch_simul(order=1, irf=40, ...).

This script is intentionally self-contained. It does not call Matlab or Dynare.
It implements:
    - calibration,
    - analytical steady state from the Matlab launcher,
    - nonlinear residuals F(x_{t-1}, x_t, E_t x_{t+1}, eps_t) = 0,
    - numerical linearization around the steady state,
    - first-order solution via a QZ/generalized-Schur decomposition,
    - impulse responses to the monetary-policy shock eR.

The first-order system has the form

    A E_t[x_{t+1}] + B x_t + C x_{t-1} + D eps_t = 0.

Given the policy rule x_t = P x_{t-1} + Q eps_t, the deterministic transition
matrix P is obtained from the stable invariant subspace of the quadratic matrix
polynomial A P^2 + B P + C = 0. The impact matrix is then

    Q = -(A P + B)^{-1} D.

Dependencies:
    numpy
    scipy
    matplotlib    only needed for plots

Example:
    python iacoviello_2005_python.py
    python iacoviello_2005_python.py --horizon 40 --shock-size 1 --save-csv
    python iacoviello_2005_python.py --shock-size 0.01

Note on shock size:
    The Dynare file sets var eR = 1 and the Taylor rule contains exp(sR * eR),
    with sR = 0.29. Therefore shock_size=1 reproduces the same one-standard-
    deviation normalization used by the supplied .mod file, implying an impact
    response of lR of approximately 0.29 log points. For smaller illustrative
    monetary-policy shocks, use e.g. --shock-size 0.01.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.linalg import ordqz


# Dynare variable ordering from the supplied .mod file.
VARS: List[str] = [
    "cp",    # patient-household consumption
    "rr",    # ex-ante real interest rate
    "b",     # debt
    "q",     # real house price
    "h",     # entrepreneur housing
    "infl",  # gross inflation
    "X",     # markup / inverse real marginal-cost object in the model
    "R",     # gross nominal interest rate
    "Y",     # output
    "c",     # entrepreneur consumption
    "hp",    # patient-household housing
    "sdf",   # stochastic discount factor used in Calvo block
    "lam",   # multiplier on borrowing constraint
    "L",     # labor
    "vp",    # price dispersion
    "w",     # real wage
    "inflp", # reset-price inflation
    "z1",    # Calvo auxiliary variable
    "z2",    # Calvo auxiliary variable
    "lR",    # log nominal rate
    "lY",    # log output
    "lpi",   # log inflation
    "lq",    # log house price
    "llam",  # log multiplier
]

IDX: Dict[str, int] = {name: i for i, name in enumerate(VARS)}


@dataclass(frozen=True)
class Params:
    """Model parameters and steady-state objects."""

    beta: float = 0.99
    gamma: float = 0.98
    nu: float = 0.03
    j: float = 0.1
    m: float = 0.89
    eta: float = 1.01
    Xs: float = 1.1
    theta: float = 0.75

    # Taylor rule and shock scaling.
    rY: float = 0.0
    rpi: float = 0.27
    rR: float = 0.73
    sR: float = 0.29

    @property
    def gammae(self) -> float:
        return self.m * self.beta + (1.0 - self.m) * self.gamma

    @property
    def kappa(self) -> float:
        return (1.0 - self.theta) * (1.0 - self.theta * self.beta) / self.theta

    @property
    def qhY(self) -> float:
        return (self.gamma * self.nu / (1.0 - self.gammae)) * (1.0 / self.Xs)

    @property
    def bY(self) -> float:
        return (
            self.beta
            * self.m
            * self.gamma
            * self.nu
            / (1.0 - self.gammae)
            * (1.0 / self.Xs)
        )

    @property
    def cY(self) -> float:
        return (
            self.nu
            / self.Xs
            * ((1.0 - self.m * self.beta) * (1.0 - self.gamma) / (1.0 - self.gammae))
        )

    @property
    def cpY(self) -> float:
        return 1.0 - self.cY

    @property
    def hH(self) -> float:
        return (1.0 + (self.j / (1.0 - self.beta)) * self.cpY / self.qhY) ** (-1.0)

    @property
    def hpH(self) -> float:
        return 1.0 - self.hH

    @property
    def hhp(self) -> float:
        return self.hH / self.hpH

    @property
    def iota(self) -> float:
        return (1.0 - self.beta) * self.hhp

    @property
    def epsi(self) -> float:
        return self.Xs / (self.Xs - 1.0)

    @property
    def rrb(self) -> float:
        return 1.0 / self.beta

    # Normalizations used in the Matlab file.
    @property
    def Ys(self) -> float:
        return 1.0

    @property
    def Hs(self) -> float:
        return 1.0

    @property
    def bs(self) -> float:
        return self.bY * self.Ys

    @property
    def cs(self) -> float:
        return self.cY * self.Ys

    @property
    def cps(self) -> float:
        return self.cpY * self.Ys

    @property
    def hs(self) -> float:
        return self.hH * self.Hs

    @property
    def hps(self) -> float:
        return self.hpH * self.Hs

    @property
    def Ls(self) -> float:
        return (((1.0 - self.nu) / self.Xs) / self.cpY) ** (1.0 / self.eta)

    @property
    def As(self) -> float:
        return 1.0 / (self.hs**self.nu * self.Ls ** (1.0 - self.nu))

    @property
    def ws(self) -> float:
        return ((1.0 - self.nu) / self.Xs) * (self.Ys / self.Ls)

    @property
    def lams(self) -> float:
        return self.beta / self.cs - self.gamma / self.cs

    @property
    def qs(self) -> float:
        return self.qhY / (self.hs * self.Ys)


@dataclass
class LinearSolution:
    """Container for the first-order approximation."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    eigenvalues: np.ndarray
    n_stable: int
    steady_state: np.ndarray


def v(x: np.ndarray, name: str) -> float:
    """Convenience accessor for a named variable."""
    return float(x[IDX[name]])


def set_v(x: np.ndarray, name: str, value: float) -> None:
    """Convenience setter for a named variable."""
    x[IDX[name]] = value


def steady_state(p: Params) -> np.ndarray:
    """Analytical steady state implied by the supplied Matlab launcher."""
    x = np.zeros(len(VARS), dtype=float)

    set_v(x, "cp", p.cps)
    set_v(x, "rr", p.rrb)
    set_v(x, "b", p.bs)
    set_v(x, "q", p.qs)
    set_v(x, "h", p.hs)
    set_v(x, "infl", 1.0)
    set_v(x, "X", p.Xs)
    set_v(x, "R", p.rrb)
    set_v(x, "Y", p.Ys)
    set_v(x, "c", p.cs)
    set_v(x, "hp", p.hps)
    set_v(x, "sdf", p.beta)
    set_v(x, "lam", p.lams)
    set_v(x, "L", p.Ls)
    set_v(x, "vp", 1.0)
    set_v(x, "w", p.ws)
    set_v(x, "inflp", 1.0)
    set_v(x, "z1", (p.Ys / p.Xs) / (1.0 - p.beta * p.theta))
    set_v(x, "z2", p.Ys / (1.0 - p.beta * p.theta))

    # Dynare's steady command makes this log(1/beta), even though the initval
    # block in the supplied .mod file starts lR at zero.
    set_v(x, "lR", np.log(p.rrb))
    set_v(x, "lY", np.log(p.Ys))
    set_v(x, "lpi", 0.0)
    set_v(x, "lq", np.log(p.qs))
    set_v(x, "llam", np.log(p.lams))
    return x


def residual(
    x_lag: np.ndarray,
    x_now: np.ndarray,
    x_lead: np.ndarray,
    eR: float,
    p: Params,
) -> np.ndarray:
    """
    Nonlinear equilibrium residuals.

    The equations are written as left-hand side minus right-hand side, matching
    the ordering of the 24 equations in the supplied Dynare .mod file.
    """
    cp = v(x_now, "cp")
    b = v(x_now, "b")
    q = v(x_now, "q")
    h = v(x_now, "h")
    infl = v(x_now, "infl")
    X = v(x_now, "X")
    R = v(x_now, "R")
    Y = v(x_now, "Y")
    c = v(x_now, "c")
    hp = v(x_now, "hp")
    sdf = v(x_now, "sdf")
    lam = v(x_now, "lam")
    L = v(x_now, "L")
    vp = v(x_now, "vp")
    w = v(x_now, "w")
    inflp = v(x_now, "inflp")
    z1 = v(x_now, "z1")
    z2 = v(x_now, "z2")
    rr = v(x_now, "rr")
    lR = v(x_now, "lR")
    lY = v(x_now, "lY")
    lpi = v(x_now, "lpi")
    lq = v(x_now, "lq")
    llam = v(x_now, "llam")

    cp_f = v(x_lead, "cp")
    c_f = v(x_lead, "c")
    q_f = v(x_lead, "q")
    L_f = v(x_lead, "L")
    X_f = v(x_lead, "X")
    infl_f = v(x_lead, "infl")
    sdf_f = v(x_lead, "sdf")
    z1_f = v(x_lead, "z1")
    z2_f = v(x_lead, "z2")

    cp_l = v(x_lag, "cp")
    b_l = v(x_lag, "b")
    h_l = v(x_lag, "h")
    infl_l = v(x_lag, "infl")
    R_l = v(x_lag, "R")
    Y_l = v(x_lag, "Y")
    vp_l = v(x_lag, "vp")

    res = np.empty(len(VARS), dtype=float)

    # (1) Housing Euler equation, patient households.
    res[0] = q / cp - p.j / hp - p.beta * q_f / cp_f

    # (2) Labor supply, patient households.
    res[1] = L ** (p.eta - 1.0) - w / cp

    # (3) Bond Euler equation, patient households.
    res[2] = 1.0 / cp - p.beta * (1.0 / cp_f) * R / infl_f

    # (4) Labor demand.
    res[3] = (1.0 - p.nu) * p.As * h_l**p.nu * L ** (-p.nu) - X * w

    # (5) Housing Euler equation, entrepreneurs.
    res[4] = (
        q / c
        - (p.gamma / c_f)
        * (p.nu * p.As * h ** (p.nu - 1.0) * L_f ** (1.0 - p.nu) / X_f + q_f)
        - p.m * lam * q_f * infl_f
    )

    # (6) Bond Euler equation, entrepreneurs.
    res[5] = 1.0 / c - p.gamma * (1.0 / c_f) * R / infl_f - lam * R

    # (7) Borrowing constraint.
    res[6] = b - p.m * (q_f * h * infl_f / R)

    # (8) Calvo auxiliary z1.
    res[7] = z1 - Y / X - p.theta * sdf_f * infl_f**p.epsi * z1_f

    # (9) Calvo auxiliary z2.
    res[8] = z2 - Y - p.theta * sdf_f * infl_f ** (p.epsi - 1.0) * z2_f

    # (10) Reset inflation.
    res[9] = inflp - (p.epsi / (p.epsi - 1.0)) * z1 / z2

    # (11) Taylor rule.
    res[10] = R - (
        p.rrb ** (1.0 - p.rR)
        * R_l**p.rR
        * (infl_l ** (1.0 + p.rpi) * (Y_l / p.Ys) ** p.rY) ** (1.0 - p.rR)
        * np.exp(p.sR * eR)
    )

    # (12) Price evolution.
    res[11] = 1.0 - p.theta * infl ** (p.epsi - 1.0) - (1.0 - p.theta) * inflp ** (1.0 - p.epsi)

    # (13) Production function.
    res[12] = Y * vp - p.As * h_l**p.nu * L ** (1.0 - p.nu)

    # (14) Price dispersion.
    res[13] = vp - (1.0 - p.theta) * inflp ** (-p.epsi) - p.theta * infl**p.epsi * vp_l

    # (15) Resource constraint.
    res[14] = c + cp - Y

    # (16) Housing equilibrium.
    res[15] = h + hp - p.Hs

    # (17) Entrepreneur budget constraint.
    res[16] = b - c - q * (h - h_l) - R_l * b_l / infl - w * L + Y * vp / X

    # (18) SDF.
    res[17] = sdf - p.beta * cp_l / cp

    # (19) Ex-ante real rate.
    res[18] = rr - R / infl_f

    # (20)-(24) Log observables / reporting variables.
    res[19] = lR - np.log(R)
    res[20] = lY - np.log(Y)
    res[21] = lpi - np.log(infl)
    res[22] = lq - np.log(q)
    res[23] = llam - np.log(lam)

    return res


def numerical_jacobian(
    p: Params,
    x_ss: np.ndarray,
    step_scale: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Central-difference linearization of F(x_{t-1}, x_t, x_{t+1}, e_t).

    Returns matrices A, B, C, D such that
        A x_{t+1} + B x_t + C x_{t-1} + D e_t = 0
    in deviations from the steady state.
    """
    n = len(VARS)
    A = np.zeros((n, n), dtype=float)
    B = np.zeros((n, n), dtype=float)
    C = np.zeros((n, n), dtype=float)
    D = np.zeros((n, 1), dtype=float)

    for j in range(n):
        step = step_scale * max(abs(x_ss[j]), 1.0)
        dx = np.zeros(n, dtype=float)
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


def _stable_selector(alpha: np.ndarray, beta: np.ndarray, stake: float) -> np.ndarray:
    """Selection rule for stable generalized eigenvalues alpha / beta."""
    eig_abs = np.full_like(np.real(alpha), fill_value=np.inf, dtype=float)
    ok = np.abs(beta) > 1e-12
    eig_abs[ok] = np.abs(alpha[ok] / beta[ok])
    return eig_abs < stake


def solve_first_order_qz(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    x_ss: np.ndarray,
    stake: float = 1.0 - 1e-8,
) -> LinearSolution:
    """
    Solve the linearized rational-expectations system with a QZ decomposition.

    For the second-order expectational difference system
        A E_t[x_{t+1}] + B x_t + C x_{t-1} + D eps_t = 0,
    form the companion pencil
        G0 z_{t+1} = G1 z_t,
    with z_t = [x_t', x_{t-1}']'. The stable invariant subspace gives
        x_t = P x_{t-1}.
    """
    n = A.shape[0]
    zeros = np.zeros((n, n), dtype=float)
    eye = np.eye(n)

    G0 = np.block([[A, zeros], [zeros, eye]])
    G1 = np.block([[-B, -C], [eye, zeros]])

    def sort_fun(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return _stable_selector(alpha, beta, stake=stake)

    # ordqz solves G1 v = lambda G0 v when called as ordqz(G1, G0).
    _, _, alpha, beta, _, Z = ordqz(G1, G0, sort=sort_fun, output="complex")

    eig = np.full(alpha.shape, np.inf + 0j, dtype=complex)
    ok = np.abs(beta) > 1e-12
    eig[ok] = alpha[ok] / beta[ok]
    n_stable = int(np.sum(np.abs(eig) < stake))

    if n_stable != n:
        raise RuntimeError(
            f"QZ solution did not find exactly n={n} stable roots. "
            f"Found {n_stable}. The Blanchard-Kahn count is not satisfied "
            "under the current ordering/tolerance."
        )

    Z_stable = Z[:, :n]
    Z_top = Z_stable[:n, :]
    Z_bottom = Z_stable[n:, :]

    if np.linalg.cond(Z_bottom) > 1e12:
        raise RuntimeError(
            "The stable invariant subspace is nearly singular. "
            "Try changing the QZ tolerance or inspecting the model timing."
        )

    P = Z_top @ np.linalg.inv(Z_bottom)
    P = np.real_if_close(P, tol=1_000).real

    AP_plus_B = A @ P + B
    Q = -np.linalg.solve(AP_plus_B, D)
    Q = np.real_if_close(Q, tol=1_000).real

    return LinearSolution(A=A, B=B, C=C, D=D, P=P, Q=Q, eigenvalues=eig, n_stable=n_stable, steady_state=x_ss)


def solve_model(p: Params, step_scale: float = 1e-6) -> LinearSolution:
    """Convenience wrapper: steady state, linearization, QZ solution."""
    x_ss = steady_state(p)
    max_ss_resid = np.max(np.abs(residual(x_ss, x_ss, x_ss, 0.0, p)))
    if max_ss_resid > 1e-7:
        raise RuntimeError(f"Steady state residuals are too large: {max_ss_resid:.3e}")

    A, B, C, D = numerical_jacobian(p, x_ss, step_scale=step_scale)
    return solve_first_order_qz(A, B, C, D, x_ss)


def impulse_response(
    sol: LinearSolution,
    horizon: int = 40,
    shock_size: float = 1.0,
) -> np.ndarray:
    """
    Generate IRFs in deviations from steady state.

    For shock_size=1, this matches the Dynare normalization var eR = 1.
    """
    n = len(VARS)
    irf = np.zeros((horizon, n), dtype=float)
    irf[0, :] = (sol.Q[:, 0] * shock_size)
    for t in range(1, horizon):
        irf[t, :] = sol.P @ irf[t - 1, :]
    return irf


def print_steady_state(p: Params, x_ss: np.ndarray) -> None:
    """Print a compact steady-state table."""
    print("\nSteady state")
    print("------------")
    for name in VARS:
        print(f"{name:>6s}: {x_ss[IDX[name]]: .10f}")


def print_solution_diagnostics(sol: LinearSolution) -> None:
    """Print basic diagnostics for the first-order solution."""
    eig_abs = np.abs(sol.eigenvalues)
    finite = np.isfinite(eig_abs)
    print("\nSolution diagnostics")
    print("--------------------")
    print(f"Stable roots selected: {sol.n_stable} out of {len(VARS)} required")
    print(f"Max |steady-state residual|: {np.max(np.abs(residual(sol.steady_state, sol.steady_state, sol.steady_state, 0.0, Params()))):.3e}")
    if np.any(finite):
        print(f"Largest finite |generalized eigenvalue|: {np.max(eig_abs[finite]):.6g}")
    quad_err = sol.A @ sol.P @ sol.P + sol.B @ sol.P + sol.C
    print(f"Max |A P^2 + B P + C|: {np.max(np.abs(quad_err)):.3e}")


def save_irfs_csv(irf: np.ndarray, output_path: Path) -> None:
    """Save IRFs as a CSV file in raw deviations from steady state."""
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["period", *VARS])
        for t in range(irf.shape[0]):
            writer.writerow([t, *irf[t, :]])
    print(f"\nSaved IRFs to: {output_path}")


def plot_irfs(irf: np.ndarray, variables: Iterable[str], scale: float = 1.0) -> None:
    """Plot selected IRFs, one figure per variable."""
    import matplotlib.pyplot as plt

    periods = np.arange(irf.shape[0])
    for name in variables:
        if name not in IDX:
            raise ValueError(f"Unknown variable {name!r}. Valid names are: {', '.join(VARS)}")
        plt.figure()
        plt.axhline(0.0, linewidth=0.8)
        plt.plot(periods, scale * irf[:, IDX[name]], linewidth=2.0)
        plt.title(f"IRF: {name}")
        plt.xlabel("Periods")
        plt.ylabel("Deviation from steady state" if scale == 1.0 else f"Deviation × {scale:g}")
        plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="First-order Python solution of the supplied Iacoviello (2005) Dynare model."
    )
    parser.add_argument("--horizon", type=int, default=40, help="IRF horizon. Default: 40.")
    parser.add_argument(
        "--shock-size",
        type=float,
        default=1.0,
        help="Size of eR shock. Default: 1, matching Dynare var eR=1.",
    )
    parser.add_argument(
        "--plot-vars",
        nargs="+",
        default=["lR", "lY", "lpi", "lq", "llam"],
        help="Variables to plot. Default: lR lY lpi lq llam.",
    )
    parser.add_argument(
        "--plot-scale",
        type=float,
        default=1.0,
        help="Multiply plotted IRFs by this number. Raw CSV is unaffected. Default: 1.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Run computations without displaying plots.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save raw IRFs to irfs_iacoviello_2005.csv.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("irfs_iacoviello_2005.csv"),
        help="Output path for CSV if --save-csv is used.",
    )
    parser.add_argument(
        "--print-steady-state",
        action="store_true",
        help="Print the full steady-state vector.",
    )
    args = parser.parse_args()

    p = Params()
    sol = solve_model(p)
    irf = impulse_response(sol, horizon=args.horizon, shock_size=args.shock_size)

    print_solution_diagnostics(sol)
    if args.print_steady_state:
        print_steady_state(p, sol.steady_state)

    print("\nSelected impact responses, period 0")
    print("-----------------------------------")
    for name in args.plot_vars:
        print(f"{name:>6s}: {irf[0, IDX[name]]: .10f}")

    if args.save_csv:
        save_irfs_csv(irf, args.csv_path)

    if not args.no_plots:
        plot_irfs(irf, args.plot_vars, scale=args.plot_scale)


if __name__ == "__main__":
    main()
