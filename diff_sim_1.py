# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: isba
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Differentiable simulation example 1: Twiss matching with four thin-lens quadrupoles
#
# Copyright (C) 2026 Hirokazu Maesaka, RIKEN SPring-8 Center. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# ## Overview
#
# A beamline consists of five drift spaces with four thin-lens quadrupoles in between:
#
# ```
# entrance -- D1 -- Q1 -- D2 -- Q2 -- D3 -- Q3 -- D4 -- Q4 -- D5 -- exit
# ```
#
# - The entrance Twiss parameters $(\beta_x, \alpha_x, \beta_y, \alpha_y)$ are drawn at random, but are treated as known.
# - The exit Twiss parameters are fixed to target values.
# - The four integrated quadrupole strengths $k_i = K_{1,i} L_i$ [1/m] are the knobs.
#
# Four knobs and four constraints make a square problem. The whole beamline is written in JAX,
# so the gradient (and the Jacobian) of the mismatch with respect to the knobs is obtained by automatic
# differentiation rather than by finite differences.
#
# ## Physics
#
# Transfer matrices (one transverse plane):
#
# $$
# M_\text{drift}(L) = \begin{pmatrix} 1 & L \\ 0 & 1 \end{pmatrix}, \qquad
# M_\text{quad}(k) = \begin{pmatrix} 1 & 0 \\ \mp k & 1 \end{pmatrix}
# $$
#
# where the upper (lower) sign is for the horizontal (vertical) plane, and $k > 0$ focuses horizontally.
#
# The beam envelope is represented by the Twiss matrix
#
# $$
# T = \begin{pmatrix} \beta & -\alpha \\ -\alpha & \gamma \end{pmatrix}, \qquad \gamma = \frac{1 + \alpha^2}{\beta},
# $$
#
# which is transported as $T_1 = M T_0 M^\mathsf{T}$.
#
# The mismatch between the obtained Twiss $(\beta, \alpha, \gamma)$ and the target $(\beta_t, \alpha_t, \gamma_t)$
# is measured by the mismatch parameter (BMAG)
#
# $$
# B = \tfrac{1}{2} \left( \beta_t \gamma - 2 \alpha_t \alpha + \gamma_t \beta \right) \ge 1,
# $$
#
# which equals 1 if and only if the two ellipses coincide. The loss function is $\mathcal{L} = (B_x - 1) + (B_y - 1)$.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use double precision; the default float32 is too coarse to reach a tight match.
jax.config.update("jax_enable_x64", True)
# %matplotlib inline

# %% [markdown]
# ## Beamline and problem setup

# %%
DRIFT_LENGTHS = jnp.array([2.0, 1.5, 1.5, 1.5, 2.0])  # [m], D1 ... D5
N_QUADS = 4

# Random (but known) entrance Twiss parameters
ENTRANCE_SEED = 0
BETA_MIN, BETA_MAX = 2.0, 15.0  # [m]
ALPHA_MIN, ALPHA_MAX = -1.5, 1.5

key = jax.random.PRNGKey(ENTRANCE_SEED)
key_beta, key_alpha = jax.random.split(key)
beta0 = jax.random.uniform(key_beta, (2,), minval=BETA_MIN, maxval=BETA_MAX)
alpha0 = jax.random.uniform(key_alpha, (2,), minval=ALPHA_MIN, maxval=ALPHA_MAX)
TWISS_IN = jnp.stack([beta0, alpha0], axis=1)  # shape (2 planes, [beta, alpha])

# Fixed exit (target) Twiss parameters
TWISS_TARGET = jnp.array(
    [
        [5.0, 0.0],  # x: beta [m], alpha
        [3.0, 0.0],  # y: beta [m], alpha
    ]
)

PLANES = ("x", "y")
for i, p in enumerate(PLANES):
    print(f"entrance {p}: beta = {TWISS_IN[i, 0]:7.4f} m, alpha = {TWISS_IN[i, 1]:+7.4f}")
for i, p in enumerate(PLANES):
    print(f"target   {p}: beta = {TWISS_TARGET[i, 0]:7.4f} m, alpha = {TWISS_TARGET[i, 1]:+7.4f}")


# %% [markdown]
# ## Forward model (pure JAX functions)
#
# Every function below is a pure function of JAX arrays, so `jax.grad`, `jax.jacfwd`, `jax.jit`, and `jax.vmap`
# can be applied freely.

# %%
def drift_matrix(length):
    return jnp.array([[1.0, length], [0.0, 1.0]])


def thin_quad_matrix(k, sign):
    """sign = +1 for x (focusing when k > 0), -1 for y."""
    return jnp.array([[1.0, 0.0], [-sign * k, 1.0]])


PLANE_SIGNS = jnp.array([1.0, -1.0])  # x, y


def twiss_to_matrix(beta, alpha):
    gamma = (1.0 + alpha**2) / beta
    return jnp.array([[beta, -alpha], [-alpha, gamma]])


def matrix_to_twiss(t):
    return jnp.array([t[0, 0], -t[0, 1]])


def transfer_matrix(k, sign):
    """One-plane transfer matrix of D1 Q1 D2 Q2 D3 Q3 D4 Q4 D5."""
    m = drift_matrix(DRIFT_LENGTHS[0])
    for i in range(N_QUADS):
        m = thin_quad_matrix(k[i], sign) @ m
        m = drift_matrix(DRIFT_LENGTHS[i + 1]) @ m
    return m


def propagate_twiss(k, twiss_in=TWISS_IN):
    """Exit Twiss parameters, shape (2 planes, [beta, alpha])."""

    def one_plane(twiss, sign):
        m = transfer_matrix(k, sign)
        t1 = m @ twiss_to_matrix(twiss[0], twiss[1]) @ m.T
        return matrix_to_twiss(t1)

    return jax.vmap(one_plane)(twiss_in, PLANE_SIGNS)


def bmag(twiss, twiss_target):
    beta, alpha = twiss
    beta_t, alpha_t = twiss_target
    gamma = (1.0 + alpha**2) / beta
    gamma_t = (1.0 + alpha_t**2) / beta_t
    return 0.5 * (beta_t * gamma - 2.0 * alpha_t * alpha + gamma_t * beta)


@jax.jit
def loss_fn(k, twiss_in=TWISS_IN):
    twiss_out = propagate_twiss(k, twiss_in)
    return jnp.sum(jax.vmap(bmag)(twiss_out, TWISS_TARGET) - 1.0)


loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

# %% [markdown]
# ### Check the automatic derivative against central finite differences

# %%
FD_STEP = 1.0e-6
k_test = jnp.array([0.3, -0.4, 0.5, -0.2])

_, grad_ad = loss_and_grad(k_test)
eye = jnp.eye(N_QUADS)
grad_fd = jnp.array([(loss_fn(k_test + FD_STEP * eye[i]) - loss_fn(k_test - FD_STEP * eye[i])) / (2.0 * FD_STEP)
                     for i in range(N_QUADS)])
print("autodiff gradient   :", np.asarray(grad_ad))
print("finite-diff gradient:", np.asarray(grad_fd))
print("max abs difference  :", float(jnp.max(jnp.abs(grad_ad - grad_fd))))

# %% [markdown]
# ## Step 1: gradient-based optimization with Adam (multi-start)
#
# The loss landscape has several local minima, so we run Adam from many random initial guesses at once.
# The optimizer loop is written with `jax.lax.scan` and vectorized over the initial guesses with `jax.vmap`;
# the whole multi-start optimization is compiled into one XLA program by `jax.jit`.

# %%
N_STARTS = 64
K_INIT_MAX = 1.0  # [1/m], initial guesses are uniform in [-K_INIT_MAX, K_INIT_MAX]
N_ADAM_STEPS = 3000
LEARNING_RATE = 1.0e-2
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1.0e-12
INIT_SEED = 1


def adam(k0, twiss_in=TWISS_IN):
    """Plain Adam written with lax.scan. Returns the final k and the loss history."""

    def step(state, i):
        k, m, v = state
        loss, g = loss_and_grad(k, twiss_in)
        m = ADAM_BETA1 * m + (1.0 - ADAM_BETA1) * g
        v = ADAM_BETA2 * v + (1.0 - ADAM_BETA2) * g**2
        m_hat = m / (1.0 - ADAM_BETA1 ** (i + 1))
        v_hat = v / (1.0 - ADAM_BETA2 ** (i + 1))
        k = k - LEARNING_RATE * m_hat / (jnp.sqrt(v_hat) + ADAM_EPS)
        return (k, m, v), loss

    state0 = (k0, jnp.zeros_like(k0), jnp.zeros_like(k0))
    (k_final, _, _), history = jax.lax.scan(step, state0, jnp.arange(N_ADAM_STEPS))
    return k_final, history


k_inits = jax.random.uniform(jax.random.PRNGKey(INIT_SEED), (N_STARTS, N_QUADS), minval=-K_INIT_MAX, maxval=K_INIT_MAX)
k_adam, loss_history = jax.jit(jax.vmap(adam))(k_inits)
loss_adam = jax.vmap(loss_fn)(k_adam)
print(f"best Adam loss over {N_STARTS} starts: {float(jnp.min(loss_adam)):.3e}")

# %% [markdown]
# ## Step 2: Newton refinement with the automatic Jacobian
#
# Near a solution, the four-residual / four-knob problem can be solved by Newton's method.
# The residual vector is $r = \left( \frac{\beta_x - \beta_{x,t}}{\beta_{x,t}},\ \alpha_x - \alpha_{x,t},\
# \frac{\beta_y - \beta_{y,t}}{\beta_{y,t}},\ \alpha_y - \alpha_{y,t} \right)$, and its 4×4 Jacobian
# $\partial r / \partial k$ is computed exactly by `jax.jacfwd`.

# %%
N_NEWTON_STEPS = 20


def residual(k, twiss_in=TWISS_IN):
    twiss_out = propagate_twiss(k, twiss_in)
    d_beta = (twiss_out[:, 0] - TWISS_TARGET[:, 0]) / TWISS_TARGET[:, 0]
    d_alpha = twiss_out[:, 1] - TWISS_TARGET[:, 1]
    return jnp.stack([d_beta, d_alpha], axis=1).ravel()


jac_fn = jax.jacfwd(residual)


def newton(k, twiss_in=TWISS_IN):
    def step(k, _):
        return k - jnp.linalg.solve(jac_fn(k, twiss_in), residual(k, twiss_in)), None

    k, _ = jax.lax.scan(step, k, None, length=N_NEWTON_STEPS)
    return k


k_newton = jax.jit(jax.vmap(newton))(k_adam)
loss_newton = jax.vmap(loss_fn)(k_newton)

# %% [markdown]
# ## Collect the distinct solutions
#
# Different starts converge to different solutions. We keep the converged ones, remove duplicates, and pick the
# solution with the smallest total quadrupole strength $\sum_i k_i^2$ as the "best" one.

# %%
CONVERGED_LOSS = 1.0e-12
DUPLICATE_TOL = 1.0e-6  # [1/m]



def distinct_solutions(k_final, loss_final):
    """Converged, de-duplicated solutions sorted by sum k^2, and the number of converged starts."""
    converged = np.asarray(jnp.isfinite(loss_final) & (loss_final < CONVERGED_LOSS))
    found = []
    for k in np.asarray(k_final)[converged]:
        if all(np.max(np.abs(k - s)) > DUPLICATE_TOL for s in found):
            found.append(k)
    return sorted(found, key=lambda s: float(np.sum(s**2))), int(converged.sum())


solutions, n_converged = distinct_solutions(k_newton, loss_newton)
print(f"{n_converged} / {N_STARTS} starts converged; {len(solutions)} distinct solutions")
print("   k1 [1/m]   k2 [1/m]   k3 [1/m]   k4 [1/m]   sum k^2")
for s in solutions:
    print(" ".join(f"{v:+10.5f}" for v in s), f"{np.sum(s**2):9.4f}")

k_best = jnp.asarray(solutions[0])
twiss_out = propagate_twiss(k_best)
print()
for i, p in enumerate(PLANES):
    print(
        f"exit {p}: beta = {twiss_out[i, 0]:.10f} m (target {TWISS_TARGET[i, 0]}), "
        f"alpha = {twiss_out[i, 1]:+.10f} (target {TWISS_TARGET[i, 1]})"
    )
print(f"loss = {float(loss_fn(k_best)):.3e}")

# %% [markdown]
# ## Plots

# %%
COLOR_X = "#2a78d6"
COLOR_Y = "#eb6834"
COLOR_MUTED = "#c3c2b7"
N_POINTS_PER_DRIFT = 50


def beta_along_line(k, twiss_in=TWISS_IN):
    """Sample s, beta_x, beta_y along the beamline."""
    s_list, beta_list = [], []
    t = jnp.stack([twiss_to_matrix(*twiss_in[i]) for i in range(2)])
    s0 = 0.0
    for i, length in enumerate(DRIFT_LENGTHS):
        for ds in jnp.linspace(0.0, length, N_POINTS_PER_DRIFT):
            m = drift_matrix(ds)
            s_list.append(s0 + ds)
            beta_list.append([(m @ t[j] @ m.T)[0, 0] for j in range(2)])
        m = drift_matrix(length)
        t = jnp.stack([m @ t[j] @ m.T for j in range(2)])
        s0 += length
        if i < N_QUADS:
            t = jnp.stack(
                [thin_quad_matrix(k[i], PLANE_SIGNS[j]) @ t[j] @ thin_quad_matrix(k[i], PLANE_SIGNS[j]).T for j in range(2)]
            )
    return np.asarray(s_list), np.asarray(beta_list)


quad_positions = np.cumsum(np.asarray(DRIFT_LENGTHS))[:N_QUADS]

# %% [markdown]
# ### Loss history of Adam (all starts)

# %%
LOSS_FLOOR = 1.0e-16  # the loss reaches round-off level (and can become slightly negative); clip for the log plot

i_best_start = int(np.argmin(np.asarray(loss_adam)))
history_clipped = np.maximum(np.asarray(loss_history), LOSS_FLOOR)
fig, ax = plt.subplots(figsize=(7, 4))
for h in history_clipped:
    ax.semilogy(h, color=COLOR_MUTED, linewidth=0.8)
ax.semilogy(history_clipped[i_best_start], color=COLOR_X, linewidth=2.0, label="best start")
ax.set_xlabel("Adam iteration")
ax.set_ylabel(r"loss $(B_x - 1) + (B_y - 1)$")
ax.set_title(f"Adam loss history ({N_STARTS} random starts)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Beta functions along the beamline: before and after matching

# %%
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for ax, k, title in zip(axes, (k_inits[i_best_start], k_best), ("initial guess", "matched")):
    s, beta = beta_along_line(k)
    ax.plot(s, beta[:, 0], color=COLOR_X, linewidth=2.0, label=r"$\beta_x$")
    ax.plot(s, beta[:, 1], color=COLOR_Y, linewidth=2.0, label=r"$\beta_y$")
    for i, p in enumerate(PLANES):
        ax.plot(s[-1], TWISS_TARGET[i, 0], "o", markersize=8, color=(COLOR_X, COLOR_Y)[i], mfc="none")
    for q in quad_positions:
        ax.axvline(q, color=COLOR_MUTED, linewidth=1.0, zorder=0)
    ax.set_ylabel(r"$\beta$ [m]")
    ax.set_title(f"{title}: k = [" + ", ".join(f"{v:+.3f}" for v in np.asarray(k)) + "] 1/m")
    ax.grid(alpha=0.3)
axes[0].legend(loc="upper left")
axes[-1].set_xlabel("s [m]  (vertical lines: quadrupoles, circles: target exit beta)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### All distinct solutions

# %%
fig, ax = plt.subplots(figsize=(8, 4))
for i, sol in enumerate(solutions):
    s, beta = beta_along_line(jnp.asarray(sol))
    lw = 2.0 if i == 0 else 1.0
    alpha_line = 1.0 if i == 0 else 0.4
    ax.plot(s, beta[:, 0], color=COLOR_X, linewidth=lw, alpha=alpha_line, label=r"$\beta_x$" if i == 0 else None)
    ax.plot(s, beta[:, 1], color=COLOR_Y, linewidth=lw, alpha=alpha_line, label=r"$\beta_y$" if i == 0 else None)
for q in quad_positions:
    ax.axvline(q, color=COLOR_MUTED, linewidth=1.0, zorder=0)
ax.set_xlabel("s [m]")
ax.set_ylabel(r"$\beta$ [m]")
ax.set_title(f"{len(solutions)} distinct matched solutions (thick: smallest $\\sum k^2$)")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Gradient-based vs. gradient-free optimization: how many simulations are needed?
#
# The main practical benefit of a differentiable simulation is that the gradient is available at the cost of
# a few forward simulations, independent of the number of knobs. Here we compare, from the same initial guesses,
#
# 1. **Nelder-Mead** (simplex search, no gradient),
# 2. **BFGS with finite-difference gradients** (gradient-based, but each gradient costs $n + 1$ simulations for
#    $n$ knobs with SciPy's default forward differences),
# 3. **BFGS with JAX gradients** (one call of `jax.value_and_grad` per evaluation).
#
# Every call of the beamline model is counted. For method 3, each call runs a forward pass and a reverse
# (backpropagation) pass. Reverse-mode automatic differentiation costs at most a small constant multiple of a
# forward pass (the "cheap gradient principle", typically a factor of 3–4 and independent of $n$), so we also
# report a conservative *forward-pass-equivalent* cost that counts each gradient call as `GRAD_COST_FACTOR` forward
# simulations.
#
# A run is counted as converged at the first simulation whose loss falls below `LOSS_TOL`.

# %%
from scipy.optimize import minimize

N_COMPARE_STARTS = 20
LOSS_TOL = 1.0e-10
MAX_EVALS = 20000
GRAD_COST_FACTOR = 4.0  # conservative cost of one forward + reverse pass in units of a forward pass

METHODS = {
    "Nelder-Mead": {"method": "Nelder-Mead", "use_jax_grad": False,
                    "options": {"maxfev": MAX_EVALS, "xatol": 1.0e-12, "fatol": 1.0e-16}},
    "BFGS (finite diff.)": {"method": "BFGS", "use_jax_grad": False,
                            "options": {"maxiter": MAX_EVALS, "gtol": 1.0e-12}},
    "BFGS (JAX grad)": {"method": "BFGS", "use_jax_grad": True,
                        "options": {"maxiter": MAX_EVALS, "gtol": 1.0e-12}},
}
COST_PER_CALL = {"Nelder-Mead": 1.0, "BFGS (finite diff.)": 1.0, "BFGS (JAX grad)": GRAD_COST_FACTOR}


def run_counted(k0, method, use_jax_grad, options):
    """Minimize the loss from k0 and record the loss of every simulation call."""
    losses = []

    def objective(k):
        if use_jax_grad:
            value, grad = loss_and_grad(jnp.asarray(k))
            losses.append(float(value))
            return float(value), np.asarray(grad)
        value = float(loss_fn(jnp.asarray(k)))
        losses.append(value)
        return value

    minimize(objective, np.asarray(k0), method=method, jac=use_jax_grad, options=options)
    return np.array(losses)


def calls_to_converge(losses):
    """Number of simulation calls until the loss first falls below LOSS_TOL (NaN if never)."""
    hit = np.flatnonzero(losses < LOSS_TOL)
    return float(hit[0] + 1) if hit.size else np.nan


histories = {name: [run_counted(k0, **cfg) for k0 in k_inits[:N_COMPARE_STARTS]] for name, cfg in METHODS.items()}
calls = {name: np.array([calls_to_converge(h) for h in hs]) for name, hs in histories.items()}


def calls_per_success(name):
    """Total calls spent (all calls for failed runs, calls to converge for successful runs) per converged run."""
    c = calls[name]
    spent = np.where(np.isfinite(c), c, [len(h) for h in histories[name]])
    return spent.sum() / max(np.isfinite(c).sum(), 1)


compare = pd.DataFrame(
    {
        "converged": [f"{np.isfinite(c).sum()} / {N_COMPARE_STARTS}" for c in calls.values()],
        "median simulation calls": [np.nanmedian(c) for c in calls.values()],
        "median forward-pass equivalent": [np.nanmedian(c) * COST_PER_CALL[n] for n, c in calls.items()],
        "forward-pass equivalent per success": [calls_per_success(n) * COST_PER_CALL[n] for n in METHODS],
        "final loss of failed runs": [
            ", ".join(f"{h.min():.3g}" for h, ci in zip(histories[n], c) if not np.isfinite(ci)) for n, c in calls.items()
        ],
    },
    index=list(METHODS),
)
compare.style.format(precision=0)

# %% [markdown]
# ### Convergence curves and cost distribution
#
# Left: best loss found so far versus the number of simulation calls, for one start. Right: number of calls needed
# to reach `LOSS_TOL` for every start (open markers for the JAX method show the forward-pass-equivalent cost).

# %%
COLOR_3 = "#1baf7a"
METHOD_COLORS = dict(zip(METHODS, (COLOR_Y, COLOR_3, COLOR_X)))
EXAMPLE_START = 0

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
for name, hs in histories.items():
    best_so_far = np.maximum(np.minimum.accumulate(hs[EXAMPLE_START]), LOSS_FLOOR)
    ax.semilogy(np.arange(1, best_so_far.size + 1), best_so_far, color=METHOD_COLORS[name], linewidth=2.0, label=name)
ax.axhline(LOSS_TOL, color=COLOR_MUTED, linewidth=1.0, linestyle="--", label="LOSS_TOL")
ax.set_xscale("log")
ax.set_xlabel("simulation calls")
ax.set_ylabel("best loss so far")
ax.set_title(f"Convergence from start {EXAMPLE_START}")
ax.legend()
ax.grid(alpha=0.3, which="both")

ax = axes[1]
for row, (name, c) in enumerate(calls.items()):
    y = np.full(c.size, row) + np.linspace(-0.15, 0.15, c.size)
    ax.plot(c, y, "o", markersize=8, color=METHOD_COLORS[name], alpha=0.7)
    ax.plot(np.nanmedian(c), row, "|", markersize=30, markeredgewidth=3.0, color="black")
    if COST_PER_CALL[name] != 1.0:
        ax.plot(c * COST_PER_CALL[name], y - 0.4, "o", markersize=8, mfc="none", color=METHOD_COLORS[name])
        ax.text(np.nanmax(c) * COST_PER_CALL[name] * 1.2, row - 0.4, "forward-pass equivalent", va="center")
ax.set_xscale("log")
ax.set_yticks(range(len(METHODS)), list(METHODS))
ax.invert_yaxis()
ax.set_xlabel(f"simulation calls to reach loss < {LOSS_TOL:.0e}  (bar: median)")
ax.set_title(f"Cost over {N_COMPARE_STARTS} random starts")
ax.grid(alpha=0.3, which="both", axis="x")
fig.tight_layout()
plt.show()

# %% [markdown]
# **Observations**
#
# - Nelder-Mead needs by far the most simulations: without gradient information it has to explore the
#   4-dimensional knob space by trial steps, and such direct-search methods typically scale poorly with the number
#   of knobs.
# - With gradients, BFGS converges in far fewer iterations. With finite-difference gradients each iteration costs
#   $n + 1$ simulations, and the attainable accuracy is limited by the difference step.
# - With JAX gradients, each iteration costs a single forward + reverse pass. Even after charging each gradient
#   call `GRAD_COST_FACTOR` forward passes, it remains much cheaper than Nelder-Mead. The advantage over finite
#   differences grows with the number of knobs, because the reverse-mode cost does not depend on $n$.
# - Some runs do not converge: they end in a *local* minimum whose loss is clearly nonzero (see the last column),
#   i.e. a quadrupole setting that is not a matching solution. A local method, with or without gradients, cannot
#   escape it, so multiple starts are needed in any case. The "per success" column charges the calls spent in
#   failed runs to the successful ones.
# - The plain Adam optimizer used earlier is *not* the most simulation-efficient gradient method for a small smooth
#   problem like this; it was used to illustrate a simple, vectorizable optimization loop. Quasi-Newton (BFGS) or
#   Newton/Gauss-Newton methods make much better use of the gradient.

# %% [markdown]
# ## Sensitivity of the loss to entrance Twiss errors
#
# In practice the entrance Twiss parameters are only known within measurement errors. We keep the matched
# quadrupole strengths $k^\ast$ fixed and ask how much the loss grows when the *actual* entrance Twiss
# parameters deviate from the assumed ones.
#
# The entrance errors are parameterized per plane as a relative beta error and an absolute alpha error,
#
# $$
# \delta = \left( \frac{\Delta\beta_x}{\beta_x},\ \Delta\alpha_x,\ \frac{\Delta\beta_y}{\beta_y},\ \Delta\alpha_y \right),
# \qquad \beta \to \beta (1 + \delta_\beta), \quad \alpha \to \alpha + \delta_\alpha .
# $$
#
# Since the loss is non-negative and vanishes at $\delta = 0$, its gradient with respect to $\delta$ is zero at the
# matched point. The leading term is therefore quadratic,
#
# $$
# \mathcal{L}(\delta) \simeq \tfrac{1}{2}\, \delta^\mathsf{T} H \delta, \qquad H = \frac{\partial^2 \mathcal{L}}{\partial \delta^2}\bigg|_{\delta = 0},
# $$
#
# and the Hessian $H$ is obtained exactly with `jax.hessian`. For Gaussian errors with covariance
# $\Sigma_\delta$, the expected loss is $\langle \mathcal{L} \rangle \simeq \tfrac{1}{2} \operatorname{tr}(H \Sigma_\delta)$.

# %%
def perturb_twiss_in(delta, twiss_in=TWISS_IN):
    """Entrance Twiss parameters with errors delta = [dbeta_x/beta_x, dalpha_x, dbeta_y/beta_y, dalpha_y]."""
    d = delta.reshape(2, 2)
    beta = twiss_in[:, 0] * (1.0 + d[:, 0])
    alpha = twiss_in[:, 1] + d[:, 1]
    return jnp.stack([beta, alpha], axis=1)


def loss_with_entrance_error(delta, k, twiss_in=TWISS_IN):
    twiss_out = propagate_twiss(k, perturb_twiss_in(delta, twiss_in))
    return jnp.sum(jax.vmap(bmag)(twiss_out, TWISS_TARGET) - 1.0)


delta_zero = jnp.zeros(2 * len(PLANES))
grad_delta = jax.grad(loss_with_entrance_error)(delta_zero, k_best)
hess_delta = jax.hessian(loss_with_entrance_error)(delta_zero, k_best)

DELTA_LABELS = ("dbeta_x/beta_x", "dalpha_x", "dbeta_y/beta_y", "dalpha_y")
print("gradient at delta = 0 (should vanish):", np.asarray(grad_delta))
print("Hessian H:")
print("".join(f"{lab:>16s}" for lab in ("",) + DELTA_LABELS))
for lab, row in zip(DELTA_LABELS, np.asarray(hess_delta)):
    print(f"{lab:>16s}" + "".join(f"{v:16.5f}" for v in row))

# %% [markdown]
# ### The loss sensitivity does not depend on the quadrupole setting
#
# The mismatch parameter $B$ is invariant under any linear symplectic transport: transporting both the actual and
# the design ellipse through the same beamline does not change their mismatch. Once the line is matched, the exit
# mismatch therefore equals the entrance mismatch, and $H$ is the same for every solution. To second order,
#
# $$
# B - 1 \simeq \tfrac{1}{2} \left[ \left( \frac{\Delta\beta}{\beta} \right)^2
# + \left( \Delta\alpha - \alpha \frac{\Delta\beta}{\beta} \right)^2 \right],
# \qquad
# H_\text{plane} = \begin{pmatrix} 1 + \alpha^2 & -\alpha \\ -\alpha & 1 \end{pmatrix},
# $$
#
# where $\alpha$ is the entrance alpha of that plane. The cell below checks this against the automatic Hessian for
# all distinct solutions.

# %%
hess_analytic = jax.scipy.linalg.block_diag(
    *[jnp.array([[1.0 + a**2, -a], [-a, 1.0]]) for a in TWISS_IN[:, 1]]
)
hess_all = jax.vmap(jax.hessian(loss_with_entrance_error), in_axes=(None, 0))(delta_zero, jnp.asarray(solutions))
for i, h in enumerate(hess_all):
    print(f"solution {i}: max |H - H_analytic| = {float(jnp.max(jnp.abs(h - hess_analytic))):.2e}")

# %% [markdown]
# ### What does depend on the solution: the exit Twiss error
#
# Although the overall mismatch is the same, the way an entrance error shows up at the exit, i.e. the
# Jacobian $\partial (\Delta\beta/\beta, \Delta\alpha)_\text{exit} / \partial \delta$, does depend on the solution.
# We summarize it with the rms exit beta-beat $\sqrt{\langle (\Delta\beta/\beta)^2 \rangle}$ for Gaussian entrance errors
# of the size below, propagated linearly with the Jacobian from `jax.jacfwd`.

# %%
SIGMA_DBETA = 0.1  # rms relative beta error at the entrance
SIGMA_DALPHA = 0.1  # rms alpha error at the entrance
COV_DELTA = jnp.diag(jnp.array([SIGMA_DBETA, SIGMA_DALPHA] * len(PLANES)) ** 2)


def exit_error(delta, k, twiss_in=TWISS_IN):
    """Exit [dbeta_x/beta_x, dalpha_x, dbeta_y/beta_y, dalpha_y] caused by entrance errors delta."""
    twiss_out = propagate_twiss(k, perturb_twiss_in(delta, twiss_in))
    d_beta = (twiss_out[:, 0] - TWISS_TARGET[:, 0]) / TWISS_TARGET[:, 0]
    d_alpha = twiss_out[:, 1] - TWISS_TARGET[:, 1]
    return jnp.stack([d_beta, d_alpha], axis=1).ravel()


jac_exit_all = jax.vmap(jax.jacfwd(exit_error), in_axes=(None, 0))(delta_zero, jnp.asarray(solutions))
expected_loss = 0.5 * jnp.trace(hess_analytic @ COV_DELTA)
print(f"expected loss 0.5 tr(H Sigma) = {float(expected_loss):.4e}  (same for all solutions)")
print("solution   sum k^2   rms dbeta/beta exit x   rms dbeta/beta exit y")
for i, (sol, jac) in enumerate(zip(solutions, jac_exit_all)):
    cov_exit = jac @ COV_DELTA @ jac.T
    rms_x, rms_y = np.sqrt(np.asarray(cov_exit)[[0, 2], [0, 2]])
    print(f"{i:8d} {np.sum(sol**2):9.4f} {rms_x:23.4f} {rms_y:23.4f}")

# %% [markdown]
# ### Monte Carlo check of the quadratic approximation
#
# Finally we sample random entrance errors, evaluate the full nonlinear loss with `jax.vmap`, and compare it with
# the quadratic prediction $\tfrac{1}{2}\delta^\mathsf{T} H \delta$. Scanning the error size shows where the
# quadratic approximation stops being accurate.

# %%
N_MC = 10000
MC_SEED = 2
# Common rms of dbeta/beta and dalpha. Keep it well below 1 so that beta*(1 + dbeta/beta) stays positive.
ERROR_SCALES = jnp.geomspace(1.0e-3, 0.25, 13)
SCATTER_SCALE = 0.2  # error size used for the scatter plot

unit_samples = jax.random.normal(jax.random.PRNGKey(MC_SEED), (N_MC, 2 * len(PLANES)))
loss_mc_fn = jax.jit(jax.vmap(loss_with_entrance_error, in_axes=(0, None)))


def quadratic_loss(delta):
    return 0.5 * jnp.einsum("ni,ij,nj->n", delta, hess_delta, delta)


mean_loss_mc = jnp.array([jnp.mean(loss_mc_fn(s * unit_samples, k_best)) for s in ERROR_SCALES])
mean_loss_quad = 0.5 * jnp.trace(hess_delta) * ERROR_SCALES**2

delta_scatter = SCATTER_SCALE * unit_samples
loss_scatter_mc = loss_mc_fn(delta_scatter, k_best)
loss_scatter_quad = quadratic_loss(delta_scatter)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.loglog(ERROR_SCALES, mean_loss_mc, "o", markersize=8, color=COLOR_X, label="Monte Carlo (full model)")
ax.loglog(ERROR_SCALES, mean_loss_quad, color=COLOR_Y, linewidth=2.0, label=r"$\frac{1}{2}\sigma^2 \mathrm{tr}\,H$")
ax.set_xlabel(r"rms entrance error $\sigma$ (both $\Delta\beta/\beta$ and $\Delta\alpha$)")
ax.set_ylabel(r"mean loss $\langle \mathcal{L} \rangle$")
ax.set_title(f"Mean loss vs. entrance error ({N_MC} samples each)")
ax.legend()
ax.grid(alpha=0.3, which="both")

ax = axes[1]
lim = float(jnp.quantile(loss_scatter_mc, 0.99))
ax.plot(loss_scatter_quad, loss_scatter_mc, ".", markersize=3, color=COLOR_X, alpha=0.3)
ax.plot([0.0, lim], [0.0, lim], color=COLOR_MUTED, linewidth=1.0, label="y = x")
ax.set_xlim(0.0, lim)
ax.set_ylim(0.0, lim)
ax.set_xlabel(r"quadratic prediction $\frac{1}{2}\delta^\mathsf{T} H \delta$")
ax.set_ylabel(r"full-model loss $\mathcal{L}(\delta)$")
ax.set_title(f"Per-sample comparison at $\\sigma$ = {SCATTER_SCALE}")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Observations**
#
# - The gradient of the loss with respect to the entrance errors vanishes at the matched point, so a first-order
#   sensitivity analysis says nothing; the Hessian is the relevant quantity.
# - The Hessian is identical for all matched solutions and agrees with the analytic formula, reflecting the
#   invariance of the mismatch parameter under linear transport. Choosing a different quadrupole solution cannot
#   make the matching more robust in terms of $B$; it only redistributes the error between $\Delta\beta/\beta$ and
#   $\Delta\alpha$ at the exit.
# - The mean loss follows $\tfrac{1}{2}\sigma^2 \operatorname{tr} H$ up to $\sigma \sim 0.1$. At larger errors the
#   per-sample scatter shows clear deviations from the quadratic model, i.e. the higher-order terms of $B$ in
#   $\Delta\beta/\beta$ become significant.

# %% [markdown]
# ## Comparison over many entrance Twiss sets
#
# So far the matching was solved for a single random entrance. We now repeat the whole procedure (multi-start Adam,
# Newton refinement, solution selection, and sensitivity analysis) for several random entrance Twiss sets and
# compare them. Case 0 is the entrance used above; the remaining cases are new random draws from the same ranges.
#
# Because every step is a pure JAX function, the extra loop over entrance cases is just one more `jax.vmap`:
# all cases × all starts are optimized in a single compiled call.

# %%
N_CASES = 10
CASES_SEED = 3

key_beta, key_alpha = jax.random.split(jax.random.PRNGKey(CASES_SEED))
new_beta = jax.random.uniform(key_beta, (N_CASES - 1, 2), minval=BETA_MIN, maxval=BETA_MAX)
new_alpha = jax.random.uniform(key_alpha, (N_CASES - 1, 2), minval=ALPHA_MIN, maxval=ALPHA_MAX)
twiss_in_cases = jnp.concatenate([TWISS_IN[None], jnp.stack([new_beta, new_alpha], axis=2)])  # (case, plane, [beta, alpha])

# vmap over starts (inner) and over entrance cases (outer); the same initial guesses are used for every case
adam_cases = jax.jit(jax.vmap(jax.vmap(adam, in_axes=(0, None)), in_axes=(None, 0)))
newton_cases = jax.jit(jax.vmap(jax.vmap(newton, in_axes=(0, None)), in_axes=(0, 0)))
loss_cases = jax.vmap(jax.vmap(loss_fn, in_axes=(0, None)), in_axes=(0, 0))

k_adam_cases, _ = adam_cases(k_inits, twiss_in_cases)
k_newton_cases = newton_cases(k_adam_cases, twiss_in_cases)
loss_newton_cases = loss_cases(k_newton_cases, twiss_in_cases)
print("k shape (case, start, quad):", k_newton_cases.shape)

# %% [markdown]
# ### Summary table
#
# For each entrance case we report
#
# - the number of starts that converged and the number of distinct solutions,
# - the smallest-$\sum k^2$ solution and the peak beta function along the line for that solution,
# - the loss sensitivity $\operatorname{tr} H$ and the expected loss $\tfrac{1}{2}\operatorname{tr}(H\Sigma_\delta)$
#   for the entrance errors $\sigma_{\Delta\beta/\beta} = $ `SIGMA_DBETA`, $\sigma_{\Delta\alpha} = $ `SIGMA_DALPHA`,
# - the range (over the distinct solutions) of the rms exit beta-beat caused by those errors.
#
# From the analytic Hessian, $\operatorname{tr} H = \sum_{x,y} (2 + \alpha^2)$, so the sensitivity depends only on the
# entrance alpha, not on the entrance beta or on the chosen solution.

# %%
hessian_fn = jax.hessian(loss_with_entrance_error)
jac_exit_fn = jax.jacfwd(exit_error)

rows = []
solutions_cases = []
for c in range(N_CASES):
    twiss_c = twiss_in_cases[c]
    sols, n_conv = distinct_solutions(k_newton_cases[c], loss_newton_cases[c])
    solutions_cases.append(sols)
    row = {
        "beta_x0": float(twiss_c[0, 0]),
        "alpha_x0": float(twiss_c[0, 1]),
        "beta_y0": float(twiss_c[1, 0]),
        "alpha_y0": float(twiss_c[1, 1]),
        "converged": n_conv,
        "n_solutions": len(sols),
    }
    if sols:
        k_c = jnp.asarray(sols[0])
        _, beta_c = beta_along_line(k_c, twiss_c)
        hess_c = hessian_fn(delta_zero, k_c, twiss_c)
        rms_beat = []
        for sol in sols:
            jac = jac_exit_fn(delta_zero, jnp.asarray(sol), twiss_c)
            cov_exit = np.asarray(jac @ COV_DELTA @ jac.T)
            rms_beat.append(np.sqrt(cov_exit[[0, 2], [0, 2]]))
        rms_beat = np.array(rms_beat)
        row |= {
            "min_sum_k2": float(np.sum(sols[0] ** 2)),
            "peak_beta_x": float(beta_c[:, 0].max()),
            "peak_beta_y": float(beta_c[:, 1].max()),
            "tr_H": float(jnp.trace(hess_c)),
            "tr_H_analytic": float(jnp.sum(2.0 + twiss_c[:, 1] ** 2)),
            "expected_loss": float(0.5 * jnp.trace(hess_c @ COV_DELTA)),
            "beat_x_min": rms_beat[:, 0].min(),
            "beat_x_max": rms_beat[:, 0].max(),
            "beat_y_min": rms_beat[:, 1].min(),
            "beat_y_max": rms_beat[:, 1].max(),
        }
    rows.append(row)

summary = pd.DataFrame(rows).rename_axis("case")
summary.style.format(precision=3)

# %% [markdown]
# ### Beta functions of the smallest-$\sum k^2$ solution for each case

# %%
N_COLS = 5
n_rows = -(-N_CASES // N_COLS)
fig, axes = plt.subplots(n_rows, N_COLS, figsize=(3.2 * N_COLS, 2.8 * n_rows), sharex=True, sharey=True)
for c, ax in enumerate(axes.flat):
    if c >= N_CASES:
        ax.set_visible(False)
        continue
    for q in quad_positions:
        ax.axvline(q, color=COLOR_MUTED, linewidth=1.0, zorder=0)
    ax.grid(alpha=0.3)
    if not solutions_cases[c]:
        ax.set_title(f"case {c}: no solution")
        continue
    s, beta = beta_along_line(jnp.asarray(solutions_cases[c][0]), twiss_in_cases[c])
    ax.plot(s, beta[:, 0], color=COLOR_X, linewidth=2.0, label=r"$\beta_x$")
    ax.plot(s, beta[:, 1], color=COLOR_Y, linewidth=2.0, label=r"$\beta_y$")
    ax.set_title(f"case {c}: {len(solutions_cases[c])} solutions")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\beta$ [m]")
for ax in axes[-1, :]:
    ax.set_xlabel("s [m]")
axes.flat[0].legend(loc="upper left")
fig.suptitle(r"Matched beta functions (smallest $\sum k^2$ solution per entrance case)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Comparison of solution properties across entrance cases

# %%
cases = summary.index.to_numpy()
fig, axes = plt.subplots(2, 2, figsize=(11, 7))

ax = axes[0, 0]
ax.bar(cases, summary["n_solutions"], color=COLOR_X)
ax.set_ylabel("distinct solutions")
ax.set_title(f"Number of distinct solutions ({N_STARTS} starts)")

ax = axes[0, 1]
ax.bar(cases, summary["min_sum_k2"], color=COLOR_X)
ax.set_ylabel(r"min $\sum k^2$ [1/m$^2$]")
ax.set_title("Weakest matching solution")

ax = axes[1, 0]
ax.vlines(cases, summary["beat_x_min"], summary["beat_x_max"], color=COLOR_X, linewidth=2.0)
ax.vlines(cases + 0.2, summary["beat_y_min"], summary["beat_y_max"], color=COLOR_Y, linewidth=2.0)
ax.plot(cases, summary["beat_x_min"], "o", markersize=8, color=COLOR_X, label="x")
ax.plot(cases + 0.2, summary["beat_y_min"], "o", markersize=8, color=COLOR_Y, label="y")
ax.set_ylabel(r"rms exit $\Delta\beta/\beta$")
ax.set_title("Exit beta-beat range over solutions (dot: minimum)")
ax.legend()

ax = axes[1, 1]
alpha_sq = summary["alpha_x0"] ** 2 + summary["alpha_y0"] ** 2
alpha_line = np.linspace(0.0, float(alpha_sq.max()), 2)
ax.plot(alpha_line, 4.0 + alpha_line, color=COLOR_MUTED, linewidth=2.0, label=r"analytic: $4 + \alpha_x^2 + \alpha_y^2$")
ax.plot(alpha_sq, summary["tr_H"], "o", markersize=8, color=COLOR_X, label="jax.hessian")
for c in cases:
    ax.annotate(str(c), (alpha_sq[c], summary["tr_H"][c]), textcoords="offset points", xytext=(5, -10))
ax.set_xlabel(r"entrance $\alpha_x^2 + \alpha_y^2$")
ax.set_ylabel(r"loss sensitivity tr $H$")
ax.set_title("Sensitivity to entrance Twiss errors")
ax.legend(loc="upper left")

for ax in axes[:, :].flat[:3]:
    ax.set_xlabel("entrance case")
    ax.set_xticks(cases)
for ax in axes.flat:
    ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Observations across entrance cases**
#
# - The number of distinct solutions and the required quadrupole strengths depend on the entrance Twiss parameters.
#   For some entrances even the weakest (smallest $\sum k^2$) solution produces a large beta function inside the line,
#   so in practice the peak beta (aperture) should be added to the objective or used as a constraint.
# - The loss sensitivity $\operatorname{tr} H$ follows $4 + \alpha_x^2 + \alpha_y^2$ exactly. A strongly converging or
#   diverging entrance beam (large $|\alpha|$) makes the matching more sensitive to Twiss measurement errors, no matter
#   which quadrupole solution is chosen.
# - The distribution of the error between $\Delta\beta/\beta$ and $\Delta\alpha$ at the exit does depend on the
#   solution, so when several solutions exist one can pick the one whose exit beta-beat is smallest.
