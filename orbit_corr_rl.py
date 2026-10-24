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
# # Reinforcement Learning for Two-Corrector Orbit Correction (NumPy DDPG)
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
# In an accelerator, the beam orbit deviates from the design orbit because of magnet misalignments, field errors,
# and injection errors. Steering magnets (*correctors*) apply small angular kicks to bring the orbit back, using
# beam position monitors (*BPMs*) as feedback. The standard method measures the **response matrix**, i.e. how the
# BPM readings change per unit corrector kick, and inverts it.
#
# In this notebook we pose the same task as a **reinforcement learning (RL)** problem: an *agent* repeatedly reads
# the BPMs, changes the corrector kicks, and receives a *reward* when the orbit improves. The agent is never told
# the optics or the incoming beam error; it must learn a correction strategy from interaction alone.
#
# The toy problem is intentionally small (two correctors, one quadrupole, two BPMs) so that
#
# - the classical solution is known exactly and serves as a reference, and
# - every part of the RL algorithm can be written in plain NumPy and inspected line by line.
#
# This notebook rewrites the original example to avoid `gymnasium`, `stable-baselines3`, and `torch`.
# We keep the same two-corrector orbit-correction task, but implement a compact actor-critic learner in the style of
# **DDPG** (Deep Deterministic Policy Gradient), with linear function approximators instead of deep networks:
#
# - NumPy for simulation, replay buffer, policy, and learning updates
# - SciPy linear algebra for stable critic fitting
# - Matplotlib for diagnostics and plots
#
# ## Notebook outline
#
# 1. Linear-optics simulator of the toy beamline
# 2. Controllability check of the response matrix
# 3. RL environment: state, action, and reward
# 4. Classical response-matrix baseline
# 5. Actor-critic components: critic, actor, replay buffer, and the nominal-model feedback as a reference
# 6. Training loop (from scratch, without prior knowledge)
# 7. Evaluation: single seed, five seeds, an example trajectory, and training variability
# 8. Discussion, limitations, and exercises
# 9. Comparison with Bayesian optimization (Gaussian process)

# %% [markdown]
# ## Learning goals
#
# After working through this notebook you should be able to:
#
# - Formulate orbit correction as a Markov decision process (state, action, reward, episode).
# - Explain why the response-matrix method solves this toy problem in one step, and what an RL agent must learn instead.
# - Describe the building blocks of an off-policy actor-critic method (DDPG): replay buffer, critic, target critic,
#   deterministic actor, and exploration noise.
# - Read a minimal NumPy implementation of these blocks and identify where it simplifies the full algorithm.
# - Evaluate a learned controller with success rate and error statistics, and recognize the limits of such evaluations.
# - Explain why training an RL agent from scratch is difficult even for a tiny linear problem, and how prior knowledge
#   (a model-based feedback) changes the picture.
# - Contrast a trained RL policy with per-episode Bayesian optimization in terms of the information and the number of
#   machine measurements each method needs.
#
# **Prerequisites:** linear transfer matrices in beam optics, basic linear algebra, and an introductory picture of RL
# (agent, environment, reward).

# %% [markdown]
# ## Setup
#
# All random numbers derive from a single seed so that the notebook is reproducible.
# The seed and most hyperparameters can be overridden by environment variables without editing the code.
# For example, in Colab run `%env SEED=29` in a cell before executing the notebook from the top.

# %%
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.stats
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


def env_float(name: str, default: float) -> float:
    """Return float environment override or default."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return float(raw_value)


def env_int(name: str, default: int) -> int:
    """Return int environment override or default."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return int(raw_value)


DEFAULT_SEED: Final = 17
SEED: Final = env_int("SEED", DEFAULT_SEED)
RNG = np.random.default_rng(SEED)

print(f"NumPy: {np.__version__}")
print(f"SciPy: {__import__('scipy').__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"Seed: {SEED}")

# %% [markdown]
# ## 1. Linear-optics simulator
#
# We track a single particle in the horizontal plane with the state vector
# $$
# u = \begin{pmatrix}x \\ x' \end{pmatrix},
# $$
# where $x$ is the position [m] and $x'$ is the angle [rad]. Each element acts on $u$ as a linear map:
#
# - Drift of length $L$:
# $$
# D(L)=\begin{pmatrix}1&L\\0&1\end{pmatrix}
# $$
# - Thin quadrupole of strength $k$:
# $$
# Q(k)=\begin{pmatrix}1&0\\-k&1\end{pmatrix}
# $$
# - Steering kick $\theta$:
# $$
# x' \leftarrow x' + \theta
# $$
#
#
# ($k>0$ is focusing.) The beamline order is
# `incoming beam -> C1 -> drift -> quad -> drift -> C2 -> drift -> BPM1 -> drift -> BPM2`:
#
# | Element | Position $s$ [m] |
# |---|---|
# | Corrector C1 | 0.00 |
# | Thin quadrupole | 0.60 |
# | Corrector C2 | 1.00 |
# | BPM1 | 1.70 |
# | BPM2 | 2.20 |
#
# **Hidden episode parameters.** At the start of every episode we draw, uniformly at random,
#
# - the incoming offset $x_0 \in [-2, 2]$ mm and angle $x'_0 \in [-1.5, 1.5]$ mrad, and
# - the quadrupole strength $k \in [-0.6, 0.6]$ m$^{-1}$.
#
# They represent an unknown injection error and an unknown optics setting (e.g. a mis-set or drifting quadrupole).
# The agent never sees them directly; it only sees BPM readings. Because $k$ changes from episode to episode,
# no single fixed correction rule derived from one response matrix is exact for all episodes. This is what makes
# the problem worth learning.

# %%
@dataclass(frozen=True)
class Lattice:
    """Geometry, hidden-latent ranges, and corrector limits for the two-corrector toy lattice."""

    drift_to_quad: float = 0.60
    drift_to_corrector_2: float = 0.40
    drift_to_bpm_1: float = 0.70
    drift_to_bpm_2: float = 0.50
    x0_limit: float = 2.0e-3
    xp0_limit: float = 1.5e-3
    k_min: float = -0.60
    k_max: float = 0.60
    max_kick: float = 4.0e-3
    max_kick_increment: float = 8.0e-4


LATTICE: Final = Lattice()


def drift(length: float) -> np.ndarray:
    """Return the 2x2 drift transfer matrix."""
    return np.array([[1.0, length], [0.0, 1.0]], dtype=np.float64)


def thin_quadrupole(strength: float) -> np.ndarray:
    """Return the 2x2 thin-quadrupole transfer matrix."""
    return np.array([[1.0, 0.0], [-strength, 1.0]], dtype=np.float64)


class BeamlineSimulator:
    """Propagate one horizontal trajectory for fixed hidden episode latents."""

    def __init__(self, lattice: Lattice) -> None:
        self.lattice = lattice
        self.x0 = 0.0
        self.xp0 = 0.0
        self.quadrupole_strength = 0.0

    def set_hidden_episode_state(self, x0: float, xp0: float, quadrupole_strength: float) -> None:
        """Set hidden episode parameters after range validation."""
        if abs(x0) > self.lattice.x0_limit or abs(xp0) > self.lattice.xp0_limit:
            raise ValueError("Incoming coordinates are outside configured limits.")
        if not self.lattice.k_min <= quadrupole_strength <= self.lattice.k_max:
            raise ValueError("Quadrupole strength is outside configured limits.")
        self.x0 = x0
        self.xp0 = xp0
        self.quadrupole_strength = quadrupole_strength

    def trajectory(self, kicks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return longitudinal positions and horizontal orbit along the beamline."""
        if kicks.shape != (2,):
            raise ValueError("Expected exactly two steering kicks.")
        state = np.array([self.x0, self.xp0 + kicks[0]], dtype=np.float64)
        positions = [0.0]
        orbit = [state[0]]

        state = drift(self.lattice.drift_to_quad) @ state
        positions.append(self.lattice.drift_to_quad)
        orbit.append(state[0])

        state = thin_quadrupole(self.quadrupole_strength) @ state
        state = drift(self.lattice.drift_to_corrector_2) @ state
        position_c2 = self.lattice.drift_to_quad + self.lattice.drift_to_corrector_2
        positions.append(position_c2)
        orbit.append(state[0])

        state[1] += kicks[1]
        state = drift(self.lattice.drift_to_bpm_1) @ state
        position_bpm1 = position_c2 + self.lattice.drift_to_bpm_1
        positions.append(position_bpm1)
        orbit.append(state[0])

        state = drift(self.lattice.drift_to_bpm_2) @ state
        positions.append(position_bpm1 + self.lattice.drift_to_bpm_2)
        orbit.append(state[0])
        return np.asarray(positions), np.asarray(orbit)

    def bpm_readings(self, kicks: np.ndarray) -> np.ndarray:
        """Return only BPM1 and BPM2 readings."""
        return self.trajectory(kicks)[1][-2:]


# %% [markdown]
# ## 2. Controllability check
#
# Because the model is linear, the BPM readings $y=(x_{\mathrm{BPM1}}, x_{\mathrm{BPM2}})^T$ are an affine function
# of the kicks $\theta=(\theta_1,\theta_2)^T$:
# $$
# y(\theta) = y_0 + R(k)\,\theta,
# $$
# where $y_0$ is the orbit produced by the incoming error $(x_0, x'_0)$ alone and $R(k)$ is the $2\times2$ response matrix.
# Writing the drift lengths as $L_1$ (C1 to quad), $L_2$ (quad to C2), $L_3$ (C2 to BPM1), and $L_4$ (BPM1 to BPM2),
# the product of the transfer matrices gives
# $$
# R(k)=
# \begin{pmatrix}
# L_1 + (L_2+L_3)(1-kL_1) & L_3 \\
# L_1 + (L_2+L_3+L_4)(1-kL_1) & L_3+L_4
# \end{pmatrix}.
# $$
#
# If $R(k)$ is invertible, the kick $\theta^\ast = -R(k)^{-1}y_0$ zeros both BPMs exactly. We scan $k$ over its full
# range and check two quantities:
#
# - $\det R$ must stay away from zero, so that a unique solution exists.
# - The smallest singular value $\sigma_{\min}$ indicates how much kick is needed in the weakest direction.
#   A small $\sigma_{\min}$ means large kicks and high sensitivity to measurement noise.

# %%
def analytic_response_matrix(quadrupole_strength: float, lattice: Lattice) -> np.ndarray:
    """Return the 2x2 analytic local response matrix."""
    q_transport = 1.0 - quadrupole_strength * lattice.drift_to_quad
    response_11 = lattice.drift_to_quad + (lattice.drift_to_corrector_2 + lattice.drift_to_bpm_1) * q_transport
    response_21 = response_11 + lattice.drift_to_bpm_2 * q_transport
    return np.array(
        [
            [response_11, lattice.drift_to_bpm_1],
            [response_21, lattice.drift_to_bpm_1 + lattice.drift_to_bpm_2],
        ],
        dtype=np.float64,
    )


k_scan = np.linspace(LATTICE.k_min, LATTICE.k_max, 401)
determinants = np.asarray([np.linalg.det(analytic_response_matrix(k_value, LATTICE)) for k_value in k_scan])
smallest_singular_values = np.asarray(
    [np.linalg.svd(analytic_response_matrix(k_value, LATTICE), compute_uv=False)[-1] for k_value in k_scan]
)
assert np.min(np.abs(determinants)) > 0.40
assert np.min(smallest_singular_values) > 0.15
print(f"Minimum |det(R)|: {np.min(np.abs(determinants)):.3f} m^2/rad^2")
print(f"Minimum singular value: {np.min(smallest_singular_values):.3f} m/rad")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(k_scan, determinants)
axes[0].axhline(0.0, color="black", linewidth=0.8)
axes[0].set(xlabel="Hidden quadrupole strength k [1/m]", ylabel="det(R) [m^2/rad^2]", title="Response determinant")
axes[1].plot(k_scan, smallest_singular_values)
axes[1].set(xlabel="Hidden quadrupole strength k [1/m]", ylabel="Smallest singular value [m/rad]", title="Conditioning")
fig.tight_layout()

# %% [markdown]
# ## 3. RL environment (without Gym)
#
# RL formalizes control as a **Markov decision process (MDP)**. At each step $t$ the agent observes a state $s_t$,
# chooses an action $a_t$, and the environment returns a reward $r_t$ and the next state $s_{t+1}$.
# The agent aims to maximize the expected discounted return
# $$
# G = \sum_{t=0}^{T-1} \gamma^t r_t,\qquad 0<\gamma<1.
# $$
#
# We implement the environment as a plain class with `reset()` and `step(action)` methods that mirror the Gymnasium API,
# so no external package is needed.
#
# - `reset()` draws new hidden parameters, sets both kicks to zero, and returns the first observation.
# - `step(action)` updates the kicks, reads the BPMs, and returns `(observation, reward, terminated, truncated, info)`.
# - An episode is **terminated** on success and **truncated** after `MAX_STEPS = 35` steps.
#   The distinction matters for the Bellman target (Section 5.1): a terminated episode has no future return,
#   whereas a truncated episode would have continued if the time limit had not stopped it.
#
# ### Observation
#
# The observation is simply the pair of BPM readings in units of $s_{\mathrm{bpm}} = 1$ mm:
# $$
# s_t = \frac{y_t}{s_{\mathrm{bpm}}} = \left(\frac{x_{\mathrm{BPM1}}}{s_{\mathrm{bpm}}},\ \frac{x_{\mathrm{BPM2}}}{s_{\mathrm{bpm}}}\right).
# $$
# Scaling to order unity helps the regression-based learners below.
#
# Note that $s_t$ does not reveal the hidden $k$, so the same reading may call for different kicks in different
# episodes: the problem is *partially observable*. A controller can still succeed by applying a correction that is
# good *on average* over $k$ and repeating it until the orbit converges. In principle the history also carries
# information about $k$, because the change of the readings caused by an action satisfies
# $\Delta y_t = \Delta\theta_{\max}R(k)\,a_{t-1}$. Exploiting this is left as an exercise (Section 8).
#
# ### Action
#
# The action is a bounded incremental command $a_t\in[-1,1]^2$, mapped to kick increments:
# $$
# \theta_{t+1} = \mathrm{clip}\left(\theta_t + \Delta\theta_{\max}\,a_t,\ -\theta_{\max},\ \theta_{\max}\right),
# \qquad \Delta\theta_{\max} = 0.8\ \mathrm{mrad},\quad \theta_{\max} = 4\ \mathrm{mrad}.
# $$
# Incremental actions mimic how operators adjust correctors in small steps and limit how far one bad action can
# push the beam. As a consequence, a large initial error usually takes several steps to correct.
#
# ### Reward
#
# The reward has two terms:
# $$
# r_t = c_{\mathrm{prog}}\left(\|s_t\|_2^2-\|s_{t+1}\|_2^2\right) + r_{\mathrm{success}},
# $$
#
# - The **progress term** ($c_{\mathrm{prog}}=0.2$) rewards every reduction of the squared orbit error. It gives dense
#   feedback at every step; a reward given only on success would be *sparse* and much harder to learn from.
#   Summed over an episode it telescopes to roughly $c_{\mathrm{prog}}(\|s_0\|^2-\|s_T\|^2)$ for $\gamma\approx1$,
#   so it guides learning without changing which final state is best.
# - The **success bonus** $r_{\mathrm{success}} = +5$ is added when $\max\lvert x_{\mathrm{BPM}}\rvert < 10$ µm,
#   which also terminates the episode. Otherwise the episode is truncated after `MAX_STEPS = 35` steps.
#
# **Why the squared norm?** With $M = (\Delta\theta_{\max}/s_{\mathrm{bpm}})R(k)$ and unclipped kicks,
# $s_{t+1} = s_t + M a_t$, so the progress term
# $$
# c_{\mathrm{prog}}\left(\|s_t\|^2-\|s_t+Ma_t\|^2\right) = -c_{\mathrm{prog}}\left(2s_t^\top M a_t + a_t^\top M^\top M a_t\right)
# $$
# is a quadratic polynomial of $(s_t, a_t)$ and is represented exactly by the quadratic critic features of Section 5.
# With the plain norm $\|s\|_2$, the reward has a cone-shaped kink at $s=0$ that a quadratic critic cannot fit
# (Exercise 4). The scale $c_{\mathrm{prog}}=0.2$ keeps per-step rewards of order ten even for the largest initial
# errors ($\|s_0\|^2$ up to about $100$). The lesson is that a reward must suit the function approximator, not only
# the task.
#
# Earlier versions of this notebook also used an action cost, one-time bonuses at intermediate error thresholds,
# and a time-out penalty. Removing them changed neither the training stability nor the final performance, so they
# were dropped. A per-step time cost, which might be expected to encourage a faster correction, did not reduce the
# number of steps either. Reward design strongly affects what an agent learns, but more terms are not automatically
# better.

# %%
BPM_SCALE: Final = 1.0e-3
MAX_KICK_INCREMENT: Final = LATTICE.max_kick_increment
MAX_KICK: Final = LATTICE.max_kick
TARGET_TOLERANCE: Final = 1.0e-5
PROGRESS_REWARD_SCALE: Final = 0.2
SUCCESS_BONUS: Final = 5.0
MAX_STEPS: Final = 35


class TwoCorrectorOrbitEnv:
    """Minimal environment API with reset and step methods."""

    def __init__(self, lattice: Lattice, rng_seed: int) -> None:
        self.lattice = lattice
        self.rng = np.random.default_rng(rng_seed)
        self.simulator = BeamlineSimulator(lattice)
        self.kicks = np.zeros(2, dtype=np.float64)
        self.normalized_bpm = np.zeros(2, dtype=np.float64)
        self.step_count = 0

    def _squared_error(self) -> float:
        return float(np.dot(self.normalized_bpm, self.normalized_bpm))

    def reset(self) -> np.ndarray:
        x0 = self.rng.uniform(-self.lattice.x0_limit, self.lattice.x0_limit)
        xp0 = self.rng.uniform(-self.lattice.xp0_limit, self.lattice.xp0_limit)
        quadrupole_strength = self.rng.uniform(self.lattice.k_min, self.lattice.k_max)
        self.simulator.set_hidden_episode_state(x0, xp0, quadrupole_strength)
        self.kicks = np.zeros(2, dtype=np.float64)
        self.normalized_bpm = self.simulator.bpm_readings(self.kicks) / BPM_SCALE
        self.step_count = 0
        return self.normalized_bpm.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply one action and return (observation, reward, terminated, truncated, info)."""
        clipped_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        previous_squared_error = self._squared_error()
        self.kicks = np.clip(
            self.kicks + self.lattice.max_kick_increment * clipped_action, -self.lattice.max_kick, self.lattice.max_kick
        )
        self.step_count += 1
        readings = self.simulator.bpm_readings(self.kicks)
        self.normalized_bpm = readings / BPM_SCALE
        max_abs_reading = float(np.max(np.abs(readings)))
        terminated = max_abs_reading < TARGET_TOLERANCE
        truncated = not terminated and self.step_count >= MAX_STEPS
        reward = PROGRESS_REWARD_SCALE * (previous_squared_error - self._squared_error())
        if terminated:
            reward += SUCCESS_BONUS
        info = {"is_success": terminated, "max_abs_reading": max_abs_reading}
        return self.normalized_bpm.copy(), reward, terminated, truncated, info


env_probe = TwoCorrectorOrbitEnv(LATTICE, rng_seed=SEED)
observation = env_probe.reset()
print("Initial observation (normalized BPM readings):", observation)
assert observation.shape == (2,)

# %% [markdown]
# ## 4. Response-matrix baseline
#
# The classical method first measures $R$ by probing each corrector around the current operating point:
# $$
# R_{:,j}\approx\frac{y(\theta+\delta e_j)-y(\theta-\delta e_j)}{2\delta},
# $$
# where $e_j$ is the unit vector for corrector $j$ and $\delta = 1$ µrad. It then applies one linear correction:
# $$
# \Delta\theta = -R^{-1}y.
# $$
#
# Because the model is linear and noiseless, a single correction reaches the target down to numerical round-off.
# This is the reference: **in this idealized setting RL cannot beat the response-matrix method.** Note, however,
# that the baseline needs four extra probe measurements per episode (since $k$ changes), whereas the RL agent uses
# only its observations. The purpose of the exercise is the methodology. Learning a controller without a model
# becomes attractive when the machine is nonlinear, noisy, or drifting, or when measuring $R$ is expensive.

# %%
RESPONSE_PROBE: Final = 1.0e-6


def estimate_response_matrix(simulator: BeamlineSimulator, kicks: np.ndarray, probe: float = RESPONSE_PROBE) -> np.ndarray:
    """Estimate the local 2x2 response matrix from symmetric kick probes."""
    response = np.empty((2, 2), dtype=np.float64)
    for column in range(2):
        positive = kicks.copy()
        negative = kicks.copy()
        positive[column] += probe
        negative[column] -= probe
        response[:, column] = (simulator.bpm_readings(positive) - simulator.bpm_readings(negative)) / (2.0 * probe)
    return response


def response_matrix_correction(environment: TwoCorrectorOrbitEnv) -> tuple[np.ndarray, np.ndarray]:
    """Apply one local response-matrix correction and return before/after BPM readings."""
    before = environment.simulator.bpm_readings(environment.kicks)
    response = estimate_response_matrix(environment.simulator, environment.kicks)
    kick_change = np.linalg.solve(response, -before)
    max_kick = environment.lattice.max_kick
    environment.kicks = np.clip(environment.kicks + kick_change, -max_kick, max_kick)
    after = environment.simulator.bpm_readings(environment.kicks)
    return before, after


baseline_env = TwoCorrectorOrbitEnv(LATTICE, rng_seed=SEED + 1)
baseline_env.reset()
baseline_before, baseline_after = response_matrix_correction(baseline_env)
print("Baseline before [mm]:", 1.0e3 * baseline_before)
print("Baseline after [um]:", 1.0e6 * baseline_after)


# %% [markdown]
# ## 5. Actor-critic components in NumPy
#
# ### 5.1 Background: deterministic policy gradient (DDPG)
#
# We use the actor-critic scheme of **DDPG** (Deep Deterministic Policy Gradient; Lillicrap et al., 2016), an
# off-policy method for continuous actions. It has two learned functions:
#
# - a **critic** $Q_w(s,a)$, which estimates the expected discounted return when taking action $a$ in state $s$ and
#   following the policy afterwards, and
# - an **actor** (policy) $\mu_W(s)$, which outputs a deterministic action.
#
# The critic is trained toward the *Bellman target*
# $$
# y = r + \gamma\,(1-\mathrm{terminated})\,Q_{\bar w}\!\left(s', \mu_W(s')\right),
# $$
# where $Q_{\bar w}$ is a slowly moving *target critic*. Only a true termination (success) cuts the future return;
# a time-limit truncation keeps the bootstrap term, because the episode would have continued.
# The actor is improved by moving its action in the direction in which the critic predicts a higher return
# (the deterministic policy gradient):
# $$
# \nabla_W J = \mathbb{E}_s\left[\nabla_a Q_w(s,a)\big|_{a=\mu_W(s)}\;\nabla_W\,\mu_W(s)\right].
# $$
# The ingredients are:
#
# | Component | Role |
# |---|---|
# | Replay buffer | Stores past transitions $(s, a, r, s', \mathrm{terminated})$ so that data can be reused (off-policy learning). |
# | Critic $Q_w$ | Estimates the return of action $a$ in state $s$; its action gradient tells the actor how to improve. |
# | Target critic $Q_{\bar w}$ | Slowly moving copy (Polyak averaging) used in the Bellman target, which stabilizes learning. |
# | Actor $\mu_W$ | Deterministic policy mapping the observation to an action in $[-1,1]^2$. |
# | Exploration noise | Gaussian noise added to the actor output while collecting data. |
#
# Modern variants such as TD3 and Soft Actor-Critic (SAC) add twin critics, target-policy smoothing, or a stochastic
# policy with an entropy bonus. We keep the plain DDPG structure (see Exercise 7).
#
# ### 5.2 Function approximators
#
# Deep RL uses neural networks for both functions. To stay NumPy-only we use simple models:
#
# 1. **Linear critic on quadratic features.** $Q_w(s,a) = w^T\varphi(s,a)$ with all monomials of $(s,a)$ up to
#    second order:
#    $$
#    \varphi(s,a) = \left[1,\ s,\ a,\ s\otimes a,\ a_1^2,\ a_2^2,\ a_1a_2,\ s_1^2,\ s_2^2,\ s_1s_2\right]
#    $$
#    (15 features for the 2-dimensional $s$ and $a$). On each mini-batch with feature matrix $\Phi$ and Bellman
#    targets $y$, the weights are moved toward the ridge-regression solution:
#    $$
#    w^\ast = (\Phi^T\Phi + \lambda I)^{-1}\Phi^T y,\qquad w \leftarrow (1-\beta)\,w + \beta\,w^\ast,
#    $$
#    with $\lambda$ = `CRITIC_RIDGE` and $\beta$ = `CRITIC_MIX`. Because $Q_w$ is quadratic in $a$, its action
#    gradient $\nabla_a Q_w$ is available in closed form.
#
#    **Why the cross terms $a_1a_2$ and $s_1s_2$?** The reward contains $a^\top M^\top M a$ (Section 3), and the
#    off-diagonal element of $M^\top M$ is about half of the diagonal ones, because both correctors move both BPMs.
#    An earlier version of this notebook omitted the two cross terms. It worked only when the actor started from
#    the nominal-model feedback (Section 5.3); training from scratch failed with it.
# 2. **Bias-free linear actor.**
#    $$
#    \mu_W(s) = \tanh(W s),
#    $$
#    with a $2\times2$ weight matrix $W$. The $\tanh$ keeps actions in $[-1,1]$. Its gradient follows from the chain
#    rule, $\nabla_W \mu_i = (1-\mu_i^2)\,s^T$, so the policy gradient on a mini-batch is
#    $$
#    g = \frac{1}{N}\sum_{n}\left[\nabla_a Q_w(s_n,a_n)\odot\left(1-a_n^2\right)\right]s_n^T,
#    \qquad a_n=\mu_W(s_n).
#    $$
#
#    **Why no bias?** Because the action is a kick *increment*, the closed loop acts as an integral controller: the
#    kicks stop moving only where the action is zero. At the target orbit $s=0$, a bias-free actor gives exactly
#    $\mu_W(0)=0$. A bias term would keep pushing the kicks at the target and leave a steady-state orbit offset of
#    tens of micrometres. This is an example of building physical knowledge into the model structure
#    (an *inductive bias*).
# 3. **Adam optimizer for the actor.** As in the original DDPG, the actor is updated with Adam
#    (Kingma and Ba, 2015):
#    $$
#    m \leftarrow \beta_1 m + (1-\beta_1) g,\quad
#    v \leftarrow \beta_2 v + (1-\beta_2) g^2,\quad
#    W \leftarrow W + \eta\,\frac{\hat m}{\sqrt{\hat v} + \epsilon},
#    $$
#    where $\hat m$ and $\hat v$ are the bias-corrected moments and $\eta$ = `ACTOR_LR`. The response matrix is
#    ill-conditioned (Section 2), so the components of $g$ differ by orders of magnitude. With plain gradient ascent
#    and a single learning rate, either the weak directions hardly move or the strong ones become unstable: in our
#    tests from $W=0$, plain gradient ascent stayed below 30% success for all but one combination of seed and
#    learning rate ($3\times10^{-4}$ to $10^{-2}$). Adam normalizes the step size of each weight.
#
# ### 5.3 Reference: nominal-model orbit feedback (prior knowledge)
#
# An accelerator physicist would first build a fixed-gain orbit feedback from the **nominal response matrix**
# $R_0 = R(k_0)$ of the design optics, here $k_0 = 0$, i.e. the ideal policy of the beamline without the quadrupole. Requiring the pre-$\tanh$ actor output to equal the full
# correction $-R_0^{-1}y/\Delta\theta_{\max}$ gives
# $$
# W_0 = -\frac{s_{\mathrm{bpm}}}{\Delta\theta_{\max}}\,R_0^{-1}.
# $$
# The $\tanh$ then saturates large corrections to the step limit, and the loop repeats the correction until the
# orbit converges. Since the actual $R(k)$ differs from $R_0$ in every episode, $W_0$ is not exact. The closed loop
# still converges, because the map $I + (\Delta\theta_{\max}/s_{\mathrm{bpm}})R(k)W_0 = I - R(k)R_0^{-1}$ is
# contracting over the whole $k$ range (its spectral radius stays below about 0.15).
#
# **By default, the RL agent starts from $W=0$, i.e. without this prior knowledge**, so that we can see what RL has
# to learn by itself and how difficult this is. $W_0$ serves as the reference that the agent should reach.
# Setting the environment variable `INITIAL_POLICY=nominal` starts the training from $W_0$ instead (*warm start*).
# Starting RL from an existing classical controller is common in accelerator applications, because the machine is
# safe from the first step. As Section 7.4 shows, this safety holds only if the policy updates are restrained
# until the critic is reliable.
#
# ### 5.4 Evaluation helper
#
# `evaluate_policy_success` runs the actor without exploration noise on fresh episodes and returns the mean return,
# the success rate, the median final $\max\lvert x_{\mathrm{BPM}}\rvert$, and the mean number of steps per episode.

# %%
def critic_features(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Create all monomials of (state, action) up to second order for the linear Q approximator."""
    if states.shape[1] != 2 or actions.shape[1] != 2:
        raise ValueError("This implementation expects exactly two states and two actions.")
    cross_terms = (states[:, :, None] * actions[:, None, :]).reshape(states.shape[0], -1)
    return np.concatenate(
        [
            np.ones((states.shape[0], 1), dtype=np.float64),
            states,
            actions,
            cross_terms,
            actions**2,
            (actions[:, 0] * actions[:, 1])[:, None],
            states**2,
            (states[:, 0] * states[:, 1])[:, None],
        ],
        axis=1,
    )


CRITIC_FEATURE_DIM: Final = 15
ACTION_START: Final = 3  # index of the linear action terms
CROSS_START: Final = 5  # index of the state-action cross terms s_i a_j (row-major in i)
ACTION_SQUARE_START: Final = 9  # index of a_1^2 and a_2^2
ACTION_PRODUCT_INDEX: Final = 11  # index of a_1 a_2


@dataclass
class ReplayBuffer:
    """Fixed-size replay buffer for off-policy updates."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    terminals: np.ndarray
    size: int = 0
    pointer: int = 0

    @classmethod
    def create(cls, capacity: int, state_dim: int, action_dim: int) -> "ReplayBuffer":
        return cls(
            states=np.zeros((capacity, state_dim), dtype=np.float64),
            actions=np.zeros((capacity, action_dim), dtype=np.float64),
            rewards=np.zeros(capacity, dtype=np.float64),
            next_states=np.zeros((capacity, state_dim), dtype=np.float64),
            terminals=np.zeros(capacity, dtype=np.float64),
        )

    @property
    def capacity(self) -> int:
        return self.states.shape[0]

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, terminated: bool) -> None:
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = next_state
        self.terminals[self.pointer] = float(terminated)
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        if self.size < batch_size:
            raise ValueError("Not enough samples in replay buffer.")
        indices = rng.integers(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.terminals[indices],
        )


@dataclass
class LinearCritic:
    """Linear Q approximator fitted by regularized least squares on each batch."""

    weights: np.ndarray

    @classmethod
    def create(cls, rng: np.random.Generator) -> "LinearCritic":
        return cls(weights=0.01 * rng.standard_normal(CRITIC_FEATURE_DIM))

    def predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return critic_features(states, actions) @ self.weights

    def action_gradient(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Return dQ/da for each state-action pair."""
        w = self.weights
        gradients = np.empty_like(actions)
        for j in range(2):
            cross_weights = w[CROSS_START + j : CROSS_START + 4 : 2]  # weights of s_1 a_j and s_2 a_j
            gradients[:, j] = (
                w[ACTION_START + j]
                + states @ cross_weights
                + 2.0 * w[ACTION_SQUARE_START + j] * actions[:, j]
                + w[ACTION_PRODUCT_INDEX] * actions[:, 1 - j]
            )
        return gradients

    def fit_batch(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray, ridge: float, mix: float) -> None:
        features = critic_features(states, actions)
        regularized = features.T @ features + ridge * np.eye(features.shape[1], dtype=np.float64)
        solved = scipy.linalg.solve(regularized, features.T @ targets, assume_a="pos")
        self.weights = (1.0 - mix) * self.weights + mix * solved


@dataclass
class LinearPolicy:
    """Deterministic bias-free linear actor a = tanh(W s), updated with Adam."""

    weight: np.ndarray
    adam_m: np.ndarray | None = None
    adam_v: np.ndarray | None = None
    adam_steps: int = 0

    def actions(self, states: np.ndarray) -> np.ndarray:
        """Return actions for a batch of states."""
        return np.tanh(states @ self.weight.T)

    def action(self, state: np.ndarray) -> np.ndarray:
        """Return the action for a single state."""
        return self.actions(state[None, :])[0]

    def noisy_action(self, state: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
        """Return an exploratory action with Gaussian noise added before the tanh."""
        pre_activation = self.weight @ state
        return np.tanh(pre_activation + noise_std * rng.standard_normal(pre_activation.shape))

    def update(
        self,
        states: np.ndarray,
        action_gradient: np.ndarray,
        learning_rate: float,
        betas: tuple[float, float],
        eps: float,
    ) -> None:
        """Take one Adam ascent step along the deterministic policy gradient."""
        actions = self.actions(states)
        gradient = (action_gradient * (1.0 - actions**2)).T @ states / states.shape[0]
        if self.adam_m is None or self.adam_v is None:
            self.adam_m = np.zeros_like(self.weight)
            self.adam_v = np.zeros_like(self.weight)
        self.adam_steps += 1
        self.adam_m = betas[0] * self.adam_m + (1.0 - betas[0]) * gradient
        self.adam_v = betas[1] * self.adam_v + (1.0 - betas[1]) * gradient**2
        m_hat = self.adam_m / (1.0 - betas[0] ** self.adam_steps)
        v_hat = self.adam_v / (1.0 - betas[1] ** self.adam_steps)
        self.weight = self.weight + learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    def copy(self) -> "LinearPolicy":
        """Return a copy of the weights (the optimizer state is not copied)."""
        return LinearPolicy(weight=self.weight.copy())


NOMINAL_QUADRUPOLE_STRENGTH: Final = 0.0


def nominal_feedback_policy(lattice: Lattice, reference_k: float = NOMINAL_QUADRUPOLE_STRENGTH) -> LinearPolicy:
    """Return the fixed-gain orbit feedback W0 = -(s_bpm / dtheta_max) R(k_ref)^-1 of the model optics."""
    nominal_response = analytic_response_matrix(reference_k, lattice)
    return LinearPolicy(weight=-(BPM_SCALE / lattice.max_kick_increment) * np.linalg.inv(nominal_response))


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate noise-free policy metrics over evaluation episodes."""

    mean_return: float
    success_rate: float
    median_error_um: float
    mean_steps: float


def run_episode(environment: TwoCorrectorOrbitEnv, policy: LinearPolicy, gamma: float = 1.0) -> tuple[float, float, dict]:
    """Run one noise-free episode from the current reset state.

    Returns the undiscounted return, the return discounted with gamma, and the last info.
    """
    state = environment.normalized_bpm.copy()
    episode_return = 0.0
    discounted_return = 0.0
    info = {"is_success": False, "max_abs_reading": float("nan")}
    terminated = truncated = False
    while not (terminated or truncated):
        discount = gamma**environment.step_count
        state, reward, terminated, truncated, info = environment.step(policy.action(state))
        episode_return += reward
        discounted_return += discount * reward
    return episode_return, discounted_return, info


def evaluate_policy_success(
    policy: LinearPolicy,
    lattice: Lattice,
    rng_seed: int,
    episodes: int,
) -> EvaluationSummary:
    """Return mean return, success rate, median final max BPM error [um], and mean episode length."""
    environment = TwoCorrectorOrbitEnv(lattice, rng_seed=rng_seed)
    returns: list[float] = []
    final_errors: list[float] = []
    episode_lengths: list[int] = []
    successes = 0
    for _ in range(episodes):
        environment.reset()
        episode_return, _, info = run_episode(environment, policy)
        returns.append(episode_return)
        final_errors.append(info["max_abs_reading"])
        episode_lengths.append(environment.step_count)
        successes += int(info["is_success"])
    return EvaluationSummary(
        mean_return=float(np.mean(returns)),
        success_rate=100.0 * successes / episodes,
        median_error_um=1.0e6 * float(np.median(final_errors)),
        mean_steps=float(np.mean(episode_lengths)),
    )


# %% [markdown]
# ## 6. Training loop
#
# Each episode has up to 35 steps and terminates early on success. Learning is off-policy: every environment step
# adds one transition to the replay buffer, and the learners are updated from random mini-batches of past transitions.
# The procedure is:
#
# 1. Initialize the actor with $W=0$ (no prior knowledge; default) or with $W_0$ (`INITIAL_POLICY=nominal`).
# 2. At every environment step:
#    1. **Act.** Use uniformly random actions during the first `WARMUP_STEPS` steps to fill the buffer with diverse
#       data; afterwards use the actor with exploration noise, $a = \tanh(Ws + \sigma\epsilon)$,
#       $\epsilon\sim\mathcal{N}(0,I)$, $\sigma$ = `EXPLORATION_STD`.
#    2. **Store** $(s, a, r, s', \mathrm{terminated})$ in the replay buffer.
#    3. **Update** on a mini-batch of `BATCH_SIZE` transitions:
#       - Compute Bellman targets $y = r + \gamma(1-\mathrm{terminated})\,Q_{\bar w}(s', \mu_W(s'))$.
#       - Refit the critic (ridge regression with partial mixing).
#       - Take one Adam step of the actor along the policy gradient (Section 5.2).
#       - Polyak-average the target critic: $\bar w \leftarrow \rho\,\bar w + (1-\rho)\,w$ with $\rho$ = `POLYAK`
#         (equivalently, a target update rate $\tau = 1-\rho = 0.01$).
# 3. Every `EVALUATION_INTERVAL_EPISODES` episodes, evaluate the actor without noise on fresh validation episodes and
#    keep the best policy (highest success rate, ties broken by the mean number of steps per episode). Stop early
#    after `EARLY_STOPPING_PATIENCE_EVALS` evaluations without improvement.
# 4. Restore the best policy.
#
# **Why keep the best policy?** RL with function approximation is often non-monotonic: a policy can improve and then
# degrade as the critic estimates shift. This happens in this notebook, too (Section 7.4). Checkpoint selection on
# validation episodes is common practice, but the final performance then partly reflects this selection.
# Validation and final evaluation therefore use different random seeds.
#
# **Why $\gamma = 0.9$?** The progress reward summed over an episode hardly depends on how fast the orbit is corrected,
# and with $\gamma$ close to 1 the discounting of the success bonus gives only a weak incentive to be fast.
# In our tests from $W=0$ over five seeds, $\gamma=0.98$ gave 7.4–10.1 steps per episode for the best policies,
# whereas $\gamma=0.9$ gave 5.9–7.1 steps and reached them in fewer episodes.
# A decaying learning rate of Adam did not prevent the degradation late in training (Section 7.4).
#
# **Simplifications relative to DDPG.** The Bellman target uses the current actor instead of a separate target
# actor, and each step performs exactly one update.
#
# Fixed configuration (after multi-seed checks):
#
# | Parameter | Value | Meaning |
# |---|---|---|
# | `GAMMA` | 0.9 | Discount factor $\gamma$ |
# | `ACTOR_LR` | $1.0\times10^{-2}$ | Adam learning rate $\eta$ of the actor |
# | `ADAM_BETAS` | (0.9, 0.999) | Adam moment decay rates $(\beta_1, \beta_2)$ |
# | `EXPLORATION_STD` | 0.2 | Standard deviation $\sigma$ of the exploration noise |
# | `CRITIC_RIDGE` | $5.0\times10^{-3}$ | Ridge penalty $\lambda$ of the critic fit |
# | `CRITIC_MIX` | 0.12 | Fraction $\beta$ of the new ridge solution mixed into the critic weights |
# | `POLYAK` | 0.99 | Target-critic averaging factor $\rho$ |
# | `BATCH_SIZE` | 256 | Transitions per update |
# | `WARMUP_STEPS` | 1000 | Random-action steps before learning starts (about 29 episodes) |
# | `TRAIN_EPISODES` | 800 | Maximum number of training episodes |
# | `EVALUATION_INTERVAL_EPISODES` | 25 | Episodes between validation runs |
# | `EARLY_STOPPING_PATIENCE_EVALS` | 12 | Validation runs without improvement before stopping |
#
# `SEED`, `TRAIN_EPISODES`, `ACTOR_LR`, `EXPLORATION_STD`, and `INITIAL_POLICY` (`zero` or `nominal`) can be
# overridden by environment variables of the same name.

# %%
GAMMA: Final = 0.9
ACTOR_LR: Final = env_float("ACTOR_LR", 1.0e-2)
ADAM_BETAS: Final = (0.9, 0.999)
ADAM_EPS: Final = 1.0e-8
EXPLORATION_STD: Final = env_float("EXPLORATION_STD", 0.2)
CRITIC_RIDGE: Final = 5.0e-3
CRITIC_MIX: Final = 0.12
POLYAK: Final = 0.99
BATCH_SIZE: Final = 256
BUFFER_CAPACITY: Final = 60_000
WARMUP_STEPS: Final = 1_000
TRAIN_EPISODES: Final = env_int("TRAIN_EPISODES", 800)
VALIDATION_EPISODES: Final = 80
EVALUATION_INTERVAL_EPISODES: Final = 25
EARLY_STOPPING_PATIENCE_EVALS: Final = 12
INITIAL_POLICY: Final = os.getenv("INITIAL_POLICY", "zero")
if INITIAL_POLICY not in ("zero", "nominal"):
    raise ValueError("INITIAL_POLICY must be 'zero' or 'nominal'.")


def initial_policy(kind: str, lattice: Lattice = LATTICE) -> LinearPolicy:
    """Return the untrained actor W = 0 ('zero') or the nominal-model feedback W0 with k0 = 0 ('nominal')."""
    if kind == "nominal":
        return nominal_feedback_policy(lattice)
    return LinearPolicy(weight=np.zeros((2, 2), dtype=np.float64))


@dataclass
class TrainingResult:
    """Outcome of one training run."""

    best_policy: LinearPolicy
    best_summary: EvaluationSummary
    best_episode: int
    episode_returns: list[float]
    validation_episodes: list[int]
    validation_success: list[float]
    validation_steps: list[float]
    replay_size: int


def train_agent(
    seed: int, kind: str, episodes: int = TRAIN_EPISODES, verbose: bool = False, lattice: Lattice = LATTICE
) -> TrainingResult:
    """Train the DDPG-style agent from the initial policy `kind` and return the best checkpoint."""
    rng = np.random.default_rng(seed)
    training_env = TwoCorrectorOrbitEnv(lattice, rng_seed=seed + 10)
    critic = LinearCritic.create(rng)
    target_critic = LinearCritic(weights=critic.weights.copy())
    replay = ReplayBuffer.create(capacity=BUFFER_CAPACITY, state_dim=2, action_dim=2)
    policy = initial_policy(kind, lattice)

    best_policy = policy.copy()
    best_summary = evaluate_policy_success(policy, lattice, rng_seed=seed + 220, episodes=VALIDATION_EPISODES)
    best_episode = 0
    validation_episodes, validation_success, validation_steps = [0], [best_summary.success_rate], [best_summary.mean_steps]
    evaluations_without_improvement = 0
    episode_returns: list[float] = []
    total_steps = 0
    for episode in range(episodes):
        state = training_env.reset()
        cumulative_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            if total_steps < WARMUP_STEPS:
                action = rng.uniform(-1.0, 1.0, size=2)
            else:
                action = policy.noisy_action(state, EXPLORATION_STD, rng)
            next_state, reward, terminated, truncated, _ = training_env.step(action)
            replay.add(state, action, reward, next_state, terminated)
            state = next_state
            cumulative_reward += reward
            total_steps += 1

            if replay.size < BATCH_SIZE or total_steps < WARMUP_STEPS:
                continue
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminals = replay.sample(BATCH_SIZE, rng)
            next_q = target_critic.predict(batch_next_states, policy.actions(batch_next_states))
            bellman_targets = batch_rewards + GAMMA * (1.0 - batch_terminals) * next_q
            critic.fit_batch(batch_states, batch_actions, bellman_targets, ridge=CRITIC_RIDGE, mix=CRITIC_MIX)
            actor_actions = policy.actions(batch_states)
            policy.update(
                batch_states, critic.action_gradient(batch_states, actor_actions), ACTOR_LR, ADAM_BETAS, ADAM_EPS
            )
            target_critic.weights = POLYAK * target_critic.weights + (1.0 - POLYAK) * critic.weights

        episode_returns.append(cumulative_reward)
        if (episode + 1) % EVALUATION_INTERVAL_EPISODES != 0:
            continue
        candidate = evaluate_policy_success(policy, lattice, rng_seed=seed + 330 + episode, episodes=VALIDATION_EPISODES)
        validation_episodes.append(episode + 1)
        validation_success.append(candidate.success_rate)
        validation_steps.append(candidate.mean_steps)
        is_better = candidate.success_rate > best_summary.success_rate or (
            np.isclose(candidate.success_rate, best_summary.success_rate) and candidate.mean_steps < best_summary.mean_steps
        )
        if is_better:
            best_summary, best_policy, best_episode = candidate, policy.copy(), episode + 1
            evaluations_without_improvement = 0
        else:
            evaluations_without_improvement += 1
        if verbose:
            print(
                f"Episode {episode + 1:4d}:",
                f"candidate success={candidate.success_rate:5.1f}%, steps={candidate.mean_steps:5.2f} |",
                f"best success={best_summary.success_rate:5.1f}%, steps={best_summary.mean_steps:5.2f},",
                f"stale_evals={evaluations_without_improvement}",
            )
        if evaluations_without_improvement >= EARLY_STOPPING_PATIENCE_EVALS:
            if verbose:
                print("Early stopping triggered by stagnant evaluation metrics.")
            break
    return TrainingResult(
        best_policy=best_policy,
        best_summary=best_summary,
        best_episode=best_episode,
        episode_returns=episode_returns,
        validation_episodes=validation_episodes,
        validation_success=validation_success,
        validation_steps=validation_steps,
        replay_size=replay.size,
    )


print(f"Training config: INITIAL_POLICY={INITIAL_POLICY}, ACTOR_LR={ACTOR_LR:.1e}, EXPLORATION_STD={EXPLORATION_STD:.2f}, GAMMA={GAMMA:.2f}")
training_result = train_agent(SEED, INITIAL_POLICY, verbose=True)
policy = training_result.best_policy
print("Training episodes:", len(training_result.episode_returns))
print("Replay size:", training_result.replay_size)
print(f"Restored best policy from episode {training_result.best_episode} for the final evaluation.")

# %% [markdown]
# ## 7. Evaluation
#
# For evaluation we switch off the exploration noise and use the actor output $a=\tanh(Ws)$ directly.
# The evaluation episodes come from a random seed not used during training or checkpoint selection. We report:
#
# - **Mean return:** the sum of the rewards. Because the progress reward telescopes, it is the same for every
#   controller that succeeds, independent of how many steps it needs.
# - **Mean discounted return** with $\gamma = 0.9$: the objective that the RL agent actually maximizes. It is higher
#   for faster corrections. Comparing it between the controllers shows whether RL found the optimum of its own objective.
# - **Success rate:** the fraction of episodes reaching $\max\lvert x_{\mathrm{BPM}}\rvert < 10$ µm within 35 steps.
# - **Mean steps per episode:** how fast the orbit is corrected. Failed episodes count as the full 35 steps.
# - **Median final error:** robust to a few diverging episodes. Because an episode ends as soon as the error falls
#   below 10 µm, this value mostly reflects how far below the threshold the last step lands.
#
# For comparison, the same numbers are shown for the nominal-model feedback $W_0$ (Section 5.3) and for repeating the
# exact correction $-R(k)^{-1}y$ with the true $k$ under the same step limit, which is the best possible controller
# with these incremental actions.

# %%
EVAL_EPISODES: Final = 150
EVAL_SEED: Final = SEED + 20


class ExactCorrectionPolicy:
    """Oracle controller that knows the hidden k and applies -R(k)^-1 y, limited to the step size."""

    def __init__(self, environment: TwoCorrectorOrbitEnv) -> None:
        self.environment = environment

    def action(self, state: np.ndarray) -> np.ndarray:
        lattice = self.environment.lattice
        response = analytic_response_matrix(self.environment.simulator.quadrupole_strength, lattice)
        kick_change = np.linalg.solve(response, -state * BPM_SCALE)
        return np.clip(kick_change / lattice.max_kick_increment, -1.0, 1.0)


def evaluate_with_errors(
    policy_factory, rng_seed: int, episodes: int, lattice: Lattice = LATTICE
) -> dict[str, np.ndarray]:
    """Run noise-free episodes and return returns, final errors, lengths, and success flags."""
    environment = TwoCorrectorOrbitEnv(lattice, rng_seed=rng_seed)
    controller = policy_factory(environment)
    results: dict[str, list] = {"returns": [], "discounted_returns": [], "final_errors": [], "lengths": [], "successes": []}
    for _ in range(episodes):
        environment.reset()
        episode_return, discounted_return, info = run_episode(environment, controller, GAMMA)
        results["returns"].append(episode_return)
        results["discounted_returns"].append(discounted_return)
        results["final_errors"].append(info["max_abs_reading"])
        results["lengths"].append(environment.step_count)
        results["successes"].append(bool(info["is_success"]))
    return {key: np.asarray(value) for key, value in results.items()}


nominal_policy = nominal_feedback_policy(LATTICE)
evaluation_results = {
    "RL policy": evaluate_with_errors(lambda env: policy, EVAL_SEED, EVAL_EPISODES),
    "Nominal feedback W0": evaluate_with_errors(lambda env: nominal_policy, EVAL_SEED, EVAL_EPISODES),
    "Exact correction (true k)": evaluate_with_errors(ExactCorrectionPolicy, EVAL_SEED, EVAL_EPISODES),
}
for name, result in evaluation_results.items():
    print(
        f"{name:26s}: return={np.mean(result['returns']):5.2f},",
        f"discounted return={np.mean(result['discounted_returns']):5.2f},",
        f"success={100.0 * np.mean(result['successes']):5.1f}%,",
        f"mean steps={np.mean(result['lengths']):5.2f},",
        f"median final error={1.0e6 * np.median(result['final_errors']):4.1f} um",
    )

# %% [markdown]
# ### 7.1 Five-seed evaluation sweep and comparison with the nominal feedback
#
# To reduce single-seed bias, we evaluate the trained deterministic policy and the nominal-model feedback $W_0$ over
# the same five independent sets of evaluation episodes. We also print how far the learned $W$ is from $W_0$.
#
# Note that these seeds change only the *evaluation episodes*; the policy comes from one training run. The spread
# therefore measures variability over test episodes, not the robustness of training (see Section 7.4 for that).

# %%
FIVE_SEED_LIST: Final[list[int]] = [17, 29, 41, 53, 67]
FIVE_SEED_EVAL_EPISODES: Final = 120
FIVE_SEED_OFFSET: Final = 50_000

for name, candidate_policy in [("RL policy", policy), ("Nominal feedback W0", nominal_policy)]:
    summaries = [
        evaluate_policy_success(candidate_policy, LATTICE, FIVE_SEED_OFFSET + benchmark_seed, FIVE_SEED_EVAL_EPISODES)
        for benchmark_seed in FIVE_SEED_LIST
    ]
    success = np.asarray([summary.success_rate for summary in summaries])
    steps = np.asarray([summary.mean_steps for summary in summaries])
    median_error = np.asarray([summary.median_error_um for summary in summaries])
    print(f"{name}:")
    for benchmark_seed, summary in zip(FIVE_SEED_LIST, summaries):
        print(
            f"  seed={benchmark_seed:2d}: success={summary.success_rate:5.1f}%,",
            f"mean steps={summary.mean_steps:5.2f}, median error={summary.median_error_um:4.1f} um",
        )
    print(
        "  aggregate:",
        f"success={np.mean(success):.1f} +/- {np.std(success):.1f}%,",
        f"steps={np.mean(steps):.2f} +/- {np.std(steps):.2f},",
        f"median error={np.mean(median_error):.2f} +/- {np.std(median_error):.2f} um",
    )

relative_distance = np.linalg.norm(policy.weight - nominal_policy.weight) / np.linalg.norm(nominal_policy.weight)
print("Learned W:\n", np.round(policy.weight, 2))
print("Nominal W0:\n", np.round(nominal_policy.weight, 2))
print(f"Relative distance |W - W0| / |W0| = {relative_distance:.2f}")

# %% [markdown]
# ### 7.2 Training curve, error distribution, and steps to success
#
# The training is shown for two initial policies, in two rows:
#
# - **Top row:** $W = 0$, i.e. no prior knowledge (the main training run of Section 6).
# - **Bottom row:** $W = W_0$, the ideal policy of the beamline **without the quadrupole** ($k = 0$, Section 5.3),
#   given to the agent as prior knowledge (warm start). If `INITIAL_POLICY=nominal` is set, the two runs swap roles.
#
# The left panels show the return per training episode and its 20-episode moving average, together with the
# validation success rate (right axis). **The first episodes (shaded) use random actions to fill the replay buffer**
# (`WARMUP_STEPS` = 1000 steps, about 29 episodes). The jump of the return at the end of this phase is therefore
# caused by the switch from random actions to the actor, not by learning. Learning shows up afterwards as the rise
# of the validation success rate. RL learning curves are noisy, and an upward trend does not guarantee that the
# latest policy is the best one (see Section 6). In the warm-started run, the validation success rate starts at
# 100% and then collapses shortly after the warm-up, before the agent relearns a policy (Section 7.4).
#
# The middle panels show the distribution of the final $\max\lvert x_{\mathrm{BPM}}\rvert$ over the evaluation
# episodes of Section 7, compared with the 10 µm success threshold, and the right panels show how many steps the
# successful evaluation episodes needed. Both use the restored best checkpoint of each run; its training episode is
# given in the panel title. For the warm start, the best checkpoint is usually $W_0$ itself or a policy from the first
# evaluation, so these panels mainly show the performance of the prior knowledge.
# Failed episodes (if any) are counted in the panel title instead of the histogram. Since every step can change each
# kick by at most $\Delta\theta_{\max}$, episodes with a large initial error need more steps even with a perfect
# controller.

# %%
def moving_average(values: list[float], window: int = 20) -> np.ndarray:
    """Return trailing moving average."""
    if len(values) < 1:
        return np.asarray([], dtype=np.float64)
    window_size = min(window, len(values))
    kernel = np.ones(window_size, dtype=np.float64) / window_size
    return np.convolve(np.asarray(values, dtype=np.float64), kernel, mode="valid")


def warmup_episodes(result: TrainingResult) -> int:
    """Return the number of episodes that were (at least partly) collected with random warm-up actions."""
    lengths = np.cumsum([MAX_STEPS] * len(result.episode_returns))
    return int(np.searchsorted(lengths, WARMUP_STEPS) + 1)


def plot_training_summary(
    axes: np.ndarray, result: TrainingResult, evaluation: dict[str, np.ndarray], label: str
) -> None:
    """Plot the training curve, the evaluation errors, and the steps to success of one training run."""
    episode_returns = result.episode_returns
    smoothed_returns = moving_average(episode_returns, window=20)
    success_mask = np.asarray(evaluation["successes"], dtype=bool)
    steps_to_success = np.asarray(evaluation["lengths"], dtype=np.int64)[success_mask]
    failed_episode_count = int(np.count_nonzero(~success_mask))
    step_bin_edges = np.arange(0.5, MAX_STEPS + 1.5, 1.0)

    axes[0].axvspan(0, warmup_episodes(result), color="gray", alpha=0.2, label="Random warm-up")
    axes[0].plot(np.arange(1, len(episode_returns) + 1), episode_returns, alpha=0.25, label="Episode return")
    axes[0].plot(np.arange(len(smoothed_returns)) + 20, smoothed_returns, linewidth=2.0, label="20-episode moving average")
    axes[0].set(xlabel="Episode", ylabel="Return", title=f"Training curve: {label}")
    success_axis = axes[0].twinx()
    success_axis.plot(result.validation_episodes, result.validation_success, "k.-", label="Validation success")
    success_axis.set(ylabel="Validation success [%]", ylim=(-5.0, 105.0))
    handles = axes[0].get_legend_handles_labels()[0] + success_axis.get_legend_handles_labels()[0]
    axes[0].legend(handles=handles, fontsize=8, loc="lower right")

    axes[1].hist(1.0e6 * np.asarray(evaluation["final_errors"]), bins=20)
    axes[1].axvline(1.0e6 * TARGET_TOLERANCE, color="tab:red", linestyle="--", label="Success threshold")
    axes[1].set(
        xlabel="Final max BPM error [um]",
        ylabel="Evaluation episodes",
        title=f"Evaluation errors (checkpoint: episode {result.best_episode})",
    )
    axes[1].legend()

    axes[2].hist(steps_to_success, bins=step_bin_edges, rwidth=0.85)
    if steps_to_success.size > 0:
        axes[2].axvline(
            np.mean(steps_to_success),
            color="tab:red",
            linestyle="--",
            label=f"Mean = {np.mean(steps_to_success):.2f}",
        )
        axes[2].set_xlim(0.5, steps_to_success.max() + 1.5)
        axes[2].legend()
    axes[2].set(
        xlabel="Steps to success",
        ylabel="Evaluation episodes",
        title=f"Steps to success (failed episodes: {failed_episode_count})",
    )


RUN_LABELS: Final = {"zero": "start from W = 0", "nominal": "start from W0 (no quadrupole)"}
other_kind = "nominal" if INITIAL_POLICY == "zero" else "zero"
other_result = train_agent(SEED, other_kind)
summary_runs = {
    INITIAL_POLICY: (training_result, evaluation_results["RL policy"]),
    other_kind: (
        other_result,
        evaluate_with_errors(lambda env: other_result.best_policy, EVAL_SEED, EVAL_EPISODES),
    ),
}

fig, axes = plt.subplots(2, 3, figsize=(16, 7.6))
for row, kind in enumerate(("zero", "nominal")):
    result, evaluation = summary_runs[kind]
    plot_training_summary(axes[row], result, evaluation, RUN_LABELS[kind])
    print(
        f"{RUN_LABELS[kind]:30s}: best checkpoint at episode {result.best_episode},",
        f"success={100.0 * np.mean(evaluation['successes']):5.1f}%, mean steps={np.mean(evaluation['lengths']):5.2f}",
    )
fig.tight_layout()

# %% [markdown]
# ### 7.3 Before/after trajectory on one held-out episode
#
# The policy observes only the two BPM readings at each step.
# The full trajectory is shown only for visualization; the agent never sees the orbit between the BPMs.
#
# Physics check: zeroing both BPMs means $x = x' = 0$ on the drift between them. Since there is no element after C2
# other than drifts, the orbit is zero all the way from C2 to BPM2. Upstream of C2 the orbit stays nonzero, because
# two BPMs can only constrain the beam downstream of the last corrector. Whether this particular episode reaches the
# 10 µm target depends on the trained policy; compare the printed final readings with the threshold.

# %%
trajectory_env = TwoCorrectorOrbitEnv(LATTICE, rng_seed=20_026)
trajectory_env.reset()
positions, orbit_before = trajectory_env.simulator.trajectory(trajectory_env.kicks)
run_episode(trajectory_env, policy)
_, orbit_after = trajectory_env.simulator.trajectory(trajectory_env.kicks)

element_positions = positions
element_names = ["Corrector 1", "Quadrupole", "Corrector 2", "BPM 1", "BPM 2"]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(positions, 1.0e3 * orbit_before, marker="o", label="Before RL correction")
ax.plot(positions, 1.0e3 * orbit_after, marker="o", label="After RL correction")
for element_position, element_name in zip(element_positions, element_names):
    ax.axvline(element_position, color="gray", linewidth=0.8, alpha=0.5)
    ax.text(element_position, ax.get_ylim()[1], element_name, rotation=90, va="top", ha="right", fontsize=8)
ax.axhline(0.0, color="black", linewidth=0.8)
ax.set(xlabel="Longitudinal position [m]", ylabel="Horizontal position x [mm]", title="Held-out trajectory")
ax.legend()
fig.tight_layout()
print(f"Steps used: {trajectory_env.step_count}")
print("Final BPM readings after RL correction [um]:", 1.0e6 * orbit_after[-2:])

# %% [markdown]
# ### 7.4 Training variability and the value of prior knowledge
#
# A single training run says little about how reliable the training is. Here the whole training is repeated with
# five training seeds, once from $W=0$ (no prior knowledge) and once from the nominal-model feedback $W_0$
# (warm start). The panels show the validation success rate and the mean number of steps during training,
# and the table lists the best checkpoint of each run. This takes about one minute.
#
# Things to look for:
#
# - From $W=0$, the success rate stays near zero for a long time and then rises within a few evaluations.
#   The episode at which this happens differs between seeds.
# - The performance can **degrade** after the best checkpoint, which is why checkpoint selection is needed.
# - From $W_0$, the policy is good from the very first episode, but shortly after the random warm-up the RL updates
#   **destroy** it: the success rate drops to almost zero while the critic is still inaccurate, and the agent then
#   relearns a slower policy. The best checkpoint is $W_0$ itself or a policy from the first evaluation.

# %%
VARIABILITY_SEEDS: Final[list[int]] = [17, 29, 41, 53, 67]

variability_results: dict[str, list[TrainingResult]] = {
    kind: [train_agent(training_seed, kind) for training_seed in VARIABILITY_SEEDS] for kind in ("zero", "nominal")
}

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for kind, line_style in [("zero", "-"), ("nominal", "--")]:
    for color_index, (training_seed, result) in enumerate(zip(VARIABILITY_SEEDS, variability_results[kind])):
        label = f"{'W=0' if kind == 'zero' else 'W0'}, seed {training_seed}"
        axes[0].plot(result.validation_episodes, result.validation_success, line_style, color=f"C{color_index}", label=label)
        axes[1].plot(result.validation_episodes, result.validation_steps, line_style, color=f"C{color_index}", label=label)
axes[0].set(xlabel="Training episode", ylabel="Validation success [%]", title="Success rate during training")
axes[1].set(xlabel="Training episode", ylabel="Mean steps per episode", title="Steps during training")
axes[1].axhline(np.mean(evaluation_results["Nominal feedback W0"]["lengths"]), color="gray", linestyle=":", label="W0 (no training)")
axes[1].legend(fontsize=7, ncol=2)
fig.tight_layout()

print(f"{'start':8s} {'seed':>4s} {'best episode':>12s} {'best success':>12s} {'best steps':>10s} {'last success':>12s} {'last steps':>10s}")
for kind in ("zero", "nominal"):
    for training_seed, result in zip(VARIABILITY_SEEDS, variability_results[kind]):
        print(
            f"{'W=0' if kind == 'zero' else 'W0':8s} {training_seed:4d} {result.best_episode:12d}",
            f"{result.best_summary.success_rate:11.1f}% {result.best_summary.mean_steps:10.2f}",
            f"{result.validation_success[-1]:11.1f}% {result.validation_steps[-1]:10.2f}",
        )

# %% [markdown]
# ### 7.5 Strong quadrupole: when the model-based feedback is wrong
#
# So far the hidden quadrupole is weak, $|k| \le 0.6$ m$^{-1}$, and the feedback $W_0$ from the model without the
# quadrupole works in every episode. Here we make the quadrupole so strong that this prior knowledge fails.
#
# **Where does $W_0$ fail?** From the response matrix of Section 2,
# $$
# \det R(k) = L_4\left[L_1 + (1-kL_1)L_2\right],
# $$
# which vanishes at $k_c = (L_1+L_2)/(L_1L_2) \approx 4.17$ m$^{-1}$: there the two correctors cannot steer the two BPMs
# independently. Without the $\tanh$ saturation, the closed loop with $W_0$ multiplies the orbit error by
# $I - R(k)R_0^{-1}$, whose nonzero eigenvalue is $k/k_c$. Hence $W_0$ diverges for $|k| \ge k_c$. For $k > k_c$ the sign
# of $\det R$ is flipped, so no rescaling $\alpha W_0$ can work: the correction points in the wrong direction.
#
# **Scenario.** We draw $k \in [5, 7]$ m$^{-1}$ (beyond $k_c$), and compare:
#
# 1. $W_0$ from the wrong model ($k_0 = 0$, as before),
# 2. $W_0$ from the correct model at a reference strength $k_\mathrm{ref} = 6$ m$^{-1}$ (center of the range),
# 3. RL trained from $W = 0$, and
# 4. RL warm-started from the wrong $W_0$ ($k_0 = 0$).
#
# Close to $k_c$, much larger kicks are needed (the smallest singular value of $R$ is about five times smaller than
# for the weak quadrupole). The kick limit and the step limit are therefore scaled by `STRONG_KICK_SCALE = 3`, i.e.
# $\theta_{\max} = 12$ mrad and $\Delta\theta_{\max} = 2.4$ mrad; the code checks that the exact correction stays within the
# kick limit. The range, the reference strength, and the scale can be changed with the environment variables
# `STRONG_K_MIN`, `STRONG_K_MAX`, `STRONG_K_REF`, and `STRONG_KICK_SCALE`. This section takes about one minute.
#
# (A thin quadrupole of $k = 6$ m$^{-1}$, i.e. a focal length of 17 cm, is unrealistically strong for this beamline;
# the scenario is a stand-in for any situation where the machine differs strongly from the model.)

# %%
STRONG_K_MIN: Final = env_float("STRONG_K_MIN", 5.0)
STRONG_K_MAX: Final = env_float("STRONG_K_MAX", 7.0)
STRONG_K_REF: Final = env_float("STRONG_K_REF", 6.0)
STRONG_KICK_SCALE: Final = env_float("STRONG_KICK_SCALE", 3.0)
STRONG_CHECK_EPISODES: Final = 2_000
STRONG_EVAL_SEED: Final = SEED + 90_000
STRONG_LATTICE: Final = Lattice(
    k_min=STRONG_K_MIN,
    k_max=STRONG_K_MAX,
    max_kick=STRONG_KICK_SCALE * LATTICE.max_kick,
    max_kick_increment=STRONG_KICK_SCALE * LATTICE.max_kick_increment,
)

critical_k = (STRONG_LATTICE.drift_to_quad + STRONG_LATTICE.drift_to_corrector_2) / (
    STRONG_LATTICE.drift_to_quad * STRONG_LATTICE.drift_to_corrector_2
)
strong_k_scan = np.linspace(STRONG_K_MIN, STRONG_K_MAX, 201)


def worst_spectral_radius(reference_k: float) -> float:
    """Return the largest spectral radius of I - R(k) R(k_ref)^-1 over the strong-k range (no saturation)."""
    reference_inverse = np.linalg.inv(analytic_response_matrix(reference_k, STRONG_LATTICE))
    return max(
        np.max(np.abs(np.linalg.eigvals(np.eye(2) - analytic_response_matrix(k, STRONG_LATTICE) @ reference_inverse)))
        for k in strong_k_scan
    )


check_env = TwoCorrectorOrbitEnv(STRONG_LATTICE, rng_seed=STRONG_EVAL_SEED + 1)
required_kicks = []
for _ in range(STRONG_CHECK_EPISODES):
    check_env.reset()
    exact_kicks = np.linalg.solve(
        analytic_response_matrix(check_env.simulator.quadrupole_strength, STRONG_LATTICE),
        -check_env.simulator.bpm_readings(np.zeros(2)),
    )
    required_kicks.append(np.max(np.abs(exact_kicks)))
required_kicks = np.asarray(required_kicks)

print(f"Critical strength k_c = {critical_k:.3f} 1/m, det R(k_c) = {np.linalg.det(analytic_response_matrix(critical_k, STRONG_LATTICE)):.1e}")
print(f"Worst spectral radius over k in [{STRONG_K_MIN}, {STRONG_K_MAX}]:",
      f"W0 from k=0: {worst_spectral_radius(0.0):.2f}, W0 from k_ref={STRONG_K_REF}: {worst_spectral_radius(STRONG_K_REF):.2f}")
print(f"Required |kick| of the exact correction [mrad]: median={1.0e3 * np.median(required_kicks):.1f},",
      f"max={1.0e3 * np.max(required_kicks):.1f}, limit={1.0e3 * STRONG_LATTICE.max_kick:.1f}")
assert np.max(required_kicks) < STRONG_LATTICE.max_kick, "Increase STRONG_KICK_SCALE: the target is not reachable."

# %%
wrong_policy = nominal_feedback_policy(STRONG_LATTICE, reference_k=0.0)
correct_policy = nominal_feedback_policy(STRONG_LATTICE, reference_k=STRONG_K_REF)
strong_runs: dict[str, list[TrainingResult]] = {
    kind: [train_agent(training_seed, kind, lattice=STRONG_LATTICE) for training_seed in VARIABILITY_SEEDS]
    for kind in ("zero", "nominal")
}


def strong_summary(name: str, policy_factory) -> tuple[float, float]:
    """Evaluate a controller on the strong-quadrupole episodes and print its success rate and mean steps."""
    result = evaluate_with_errors(policy_factory, STRONG_EVAL_SEED, EVAL_EPISODES, lattice=STRONG_LATTICE)
    success, steps = 100.0 * np.mean(result["successes"]), np.mean(result["lengths"])
    print(f"  {name:44s}: success={success:5.1f}%, mean steps={steps:5.2f}")
    return success, steps


print(f"Strong quadrupole, k in [{STRONG_K_MIN}, {STRONG_K_MAX}] 1/m, {EVAL_EPISODES} evaluation episodes:")
strong_summary("Exact correction (true k)", ExactCorrectionPolicy)
strong_summary("W0 from the wrong model (k0 = 0)", lambda env: wrong_policy)
correct_steps = strong_summary(f"W0 from the correct model (k_ref = {STRONG_K_REF})", lambda env: correct_policy)[1]
for kind, label in [("zero", "RL from W = 0"), ("nominal", "RL from the wrong W0")]:
    for training_seed, result in zip(VARIABILITY_SEEDS, strong_runs[kind]):
        strong_summary(f"{label}, seed {training_seed} (best episode {result.best_episode})", lambda env: result.best_policy)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for kind, line_style, label in [("zero", "-", "W=0"), ("nominal", "--", "wrong W0")]:
    for color_index, (training_seed, result) in enumerate(zip(VARIABILITY_SEEDS, strong_runs[kind])):
        axes[0].plot(result.validation_episodes, result.validation_success, line_style, color=f"C{color_index}",
                     label=f"{label}, seed {training_seed}")
        axes[1].plot(result.validation_episodes, result.validation_steps, line_style, color=f"C{color_index}",
                     label=f"{label}, seed {training_seed}")
axes[1].axhline(correct_steps, color="gray", linestyle=":", label=f"W0 from k_ref = {STRONG_K_REF} (no training)")
axes[0].set(xlabel="Training episode", ylabel="Validation success [%]", title="Strong quadrupole: success rate")
axes[1].set(xlabel="Training episode", ylabel="Mean steps per episode", title="Strong quadrupole: steps")
axes[1].legend(fontsize=7, ncol=2)
fig.tight_layout()

# %% [markdown]
# The panels below show the same diagnostics as Section 7.2 for the strong quadrupole, for the training seed `SEED`:
# the top row starts from $W = 0$, the bottom row from the wrong $W_0$ ($k_0 = 0$). As in Section 7.2, the evaluation
# panels use the restored best checkpoint of each run on the strong-quadrupole evaluation episodes.

# %%
STRONG_RUN_LABELS: Final = {"zero": "strong k, start from W = 0", "nominal": "strong k, start from wrong W0 (k0 = 0)"}


def strong_run_for_seed(kind: str) -> TrainingResult:
    """Return the strong-quadrupole training run with the training seed SEED (reused from above if available)."""
    if SEED in VARIABILITY_SEEDS:
        return strong_runs[kind][VARIABILITY_SEEDS.index(SEED)]
    return train_agent(SEED, kind, lattice=STRONG_LATTICE)


fig, axes = plt.subplots(2, 3, figsize=(16, 7.6))
for row, kind in enumerate(("zero", "nominal")):
    result = strong_run_for_seed(kind)
    evaluation = evaluate_with_errors(
        lambda env: result.best_policy, STRONG_EVAL_SEED, EVAL_EPISODES, lattice=STRONG_LATTICE
    )
    plot_training_summary(axes[row], result, evaluation, STRONG_RUN_LABELS[kind])
    print(
        f"{STRONG_RUN_LABELS[kind]:40s}: best checkpoint at episode {result.best_episode},",
        f"success={100.0 * np.mean(evaluation['successes']):5.1f}%, mean steps={np.mean(evaluation['lengths']):5.2f}",
    )
fig.tight_layout()

# %% [markdown]
# Things to look for:
#
# - The feedback from the wrong model fails in every episode, while the feedback from the correct model works.
#   A model-based controller is only as good as its model.
# - RL needs no model. Trained from $W = 0$ or from the wrong $W_0$, it learns to correct the orbit in (almost) every
#   episode within a few hundred training episodes, where the wrong prior knowledge gives nothing.
# - RL does not reach the feedback from the correct model, for the same reason as in the weak-quadrupole case
#   (the approximation error of the critic, Section 8).
# - In short: RL can beat a wrong model, but not a correct one. In practice, the best of both is to keep the model
#   up to date (e.g. by measuring the response matrix) and to use RL for what the model cannot describe.

# %% [markdown]
# ## 8. Discussion and limitations
#
# **RL versus the response-matrix method.** In this linear, noiseless model, the response-matrix correction is exact
# after a few probe measurements, while the RL agent needs tens of thousands of interaction steps and converges over
# several steps. RL becomes attractive when the system is nonlinear, noisy, or drifting, when constraints matter, or
# when measuring the response is costly. Even then, data efficiency and machine safety are major concerns: agents
# are usually trained in simulation and then transferred to the real machine (*sim-to-real*), often combined with
# model-based methods.
#
# **Training from scratch is difficult, even for this tiny problem.** The task has only four policy parameters and a
# linear plant, yet training from $W=0$ needed several design decisions:
#
# - a reward that the critic can represent (the squared error, Section 3),
# - a critic with all second-order terms, including $a_1a_2$ and $s_1s_2$ (Section 5.2),
# - a bias-free actor, so that the action vanishes at the target orbit (Section 5.2),
# - Adam instead of plain gradient ascent, because the problem is ill-conditioned (Section 5.2),
# - a smaller discount factor $\gamma = 0.9$ to reward a fast correction (Section 6), and
# - checkpoint selection, because the policy degrades later in training (Section 7.4).
#
# In our multi-seed tests, training from $W=0$ failed without the cross terms of the critic or without Adam, and
# $\gamma = 0.98$ gave slower policies. Even with all of the above, the learned policy needs about twice as many
# steps as the nominal-model feedback $W_0$ (Sections 7 and 7.1), and the training outcome depends on the seed
# (Section 7.4).
#
# **The critic, not the reward, limits the result.** The discounted return of Section 7, which is exactly the
# objective of the RL agent, is higher for $W_0$ than for the learned policy. In a separate test, scaling $W_0$ by
# 0.5 or 1.5 also lowered it, so $W_0$ is close to the optimum of the RL objective among linear gains.
# The agent nevertheless converges elsewhere, and even a warm start from $W_0$ is first destroyed and then replaced by
# a slower policy (Section 7.4).
# The reason is the approximation error of the quadratic critic: it fits the one-step reward exactly, but not the
# effect of the $\tanh$ saturation, the kick limits, and the termination on success on the future return. Its action
# gradient therefore points the actor in a slightly wrong direction. This is a small-scale example of a central
# difficulty of RL with function approximation: errors of the learned value function are exploited by the policy.
#
# **The value of prior knowledge.** The nominal-model feedback $W_0$ costs no RL interaction at all, because it
# follows from the lattice model or from one response-matrix measurement, and it already succeeds in every episode
# with fewer steps than the trained policy. Here, the most useful role of RL training would be to adapt to what the
# model does not know. The remaining gap to the exact correction (Section 7) comes from the unknown $k$. Closing it
# requires a policy that adapts to $k$, which in turn requires history in the observation (Exercise 3).
#
# **When the model is wrong.** With a strong quadrupole (Section 7.5), the feedback from the model without the
# quadrupole fails completely, and RL learns a working controller from scratch or from the wrong prior. A feedback from
# the correct model is still better. The value of RL is therefore largest where no good model is available.
#
# **What was simplified, and why.** An earlier, SAC-style version of this notebook was reduced step by step, and each
# reduction was checked over several training seeds: removing the action cost, the milestone bonuses, and the time-out
# penalty from the reward; shrinking the observation to the BPM readings; replacing twin critics by a single critic;
# removing the stochastic policy and entropy bonus of SAC; and replacing a small neural-network actor by a linear one.
#
# **Implementation limitations.**
#
# - This is a compact educational implementation of DDPG-style learning, not a full deep-RL stack.
# - The critic is linear in hand-crafted features. They fit the one-step reward exactly, but the true $Q$ function
#   also contains the future return and the effect of the action and kick limits, which a quadratic cannot represent.
# - The checkpoint is selected on 80 validation episodes. Small differences in the number of steps are partly noise
#   of this selection.
#
# **Model limitations.**
#
# - The machine model is linear, noiseless, and horizontal-only.
# - For more realism, add noise, calibration drift, constraints, and temporal history.
#
# ## Exercises
#
# 1. **Warm start.** Run the notebook with `%env INITIAL_POLICY=nominal`. How do the learning curve, the final number of
#    steps, and the learned $W$ compare with the training from $W=0$? Can you keep the warm-started policy safe, e.g.
#    by updating only the critic for the first few hundred steps, or by a smaller `ACTOR_LR`?
# 2. **Training variability.** Extend `VARIABILITY_SEEDS` in Section 7.4 and report the mean and spread of the best
#    and the last success rate. How often does the policy degrade after its best checkpoint?
# 3. **Using history.** Add the BPM change $\Delta s_t$ and the previous action to the observation, and extend the
#    critic features accordingly. Can the agent learn to adapt to the hidden $k$ and need fewer steps?
# 4. **Reward design.** Replace the squared-error progress term by the plain norm $\|s\|_2$. What happens to the
#    critic's action gradient and to the policy during training?
# 5. **Critic features and optimizer.** Remove the cross terms $a_1a_2$ and $s_1s_2$ from `critic_features`, or replace
#    the Adam step by a plain gradient step $W \leftarrow W + \eta g$. Can the agent still learn from $W=0$?
#    Conversely, can a richer critic (e.g. features of the pre-$\tanh$ action, or a small neural network) bring the
#    learned policy closer to $W_0$?
# 6. **Exploration and discounting.** Vary `EXPLORATION_STD` (e.g. `%env EXPLORATION_STD=0.5`) and `GAMMA`.
#    How do they affect the speed and the stability of training?
# 7. **Toward TD3 or SAC.** Add a second critic and use the minimum of both in the Bellman target (TD3), or make the
#    policy stochastic with an entropy bonus (SAC). Does either help in this problem?
# 8. **Measurement noise.** Add Gaussian BPM noise of a few micrometres in `bpm_readings`. How do the RL agent and the
#    response-matrix baseline degrade? (Hint: the 1 µrad probe of the baseline becomes too small.)

# %% [markdown]
# ## 9. Comparison with Bayesian optimization (Gaussian process)
#
# A widely used alternative to RL for online accelerator tuning is **Bayesian optimization (BO)**. Instead of
# learning a policy in advance, BO tunes each new situation from scratch: it treats the machine as a black-box function
# of the knobs and chooses each new setting by balancing what it has already learned against what it has not
# explored yet.
#
# ### 9.1 Problem setting
#
# For every episode (fixed but unknown $x_0$, $x'_0$, $k$) we minimize the scalar orbit objective
# $$
# f(\theta) = \left\|\frac{y(\theta)}{s_{\mathrm{bpm}}}\right\|_2^2
# $$
# over the kicks $\theta\in[-\theta_{\max},\theta_{\max}]^2$. Each evaluation of $f$ means setting the correctors and
# reading the BPMs once, i.e. one **machine measurement**, which corresponds to one RL step. BO starts from the reading
# at $\theta=0$, which all methods get for free at reset, and stops as soon as an evaluated setting satisfies
# $\max\lvert x_{\mathrm{BPM}}\rvert<10$ µm, or after `MAX_STEPS` evaluations.
#
# ### 9.2 Gaussian-process surrogate
#
# Given evaluated points $X=\{\theta_i\}$ with values $F=\{f_i\}$, a Gaussian process with the kernel
# $$
# K(\theta,\theta') = c\,\exp\left(-\frac{\|\theta-\theta'\|^2}{2\ell^2}\right)
# $$
# and measurement noise of variance $\sigma_n^2$ gives the posterior mean and variance at a new point $\theta$:
# $$
# \mu(\theta) = k_\theta^T\left(\mathbf{K} + \sigma_n^2 I\right)^{-1}F,\qquad
# \sigma^2(\theta) = K(\theta,\theta) - k_\theta^T\left(\mathbf{K} + \sigma_n^2 I\right)^{-1}k_\theta,
# $$
# where $\mathbf{K} = (K(\theta_i,\theta_j))_{ij}$ and $k_\theta = (K(\theta,\theta_i))_i$. In scikit-learn the noise is
# part of the kernel object, `ConstantKernel * RBF + WhiteKernel`, where `WhiteKernel` adds $\sigma_n^2$ to the
# diagonal of $\mathbf{K}$. The kicks are normalized by $\theta_{\max}$, and the hyperparameters $c$, $\ell$, and
# $\sigma_n$ are refitted by maximum marginal likelihood after every new measurement.
#
# ### 9.3 Acquisition function: Expected Improvement
#
# The next setting maximizes the **Expected Improvement** over the best value so far, $f^\ast=\min_i f_i$:
# $$
# \mathrm{EI}(\theta) = \left(f^\ast-\mu(\theta)\right)\Phi(z) + \sigma(\theta)\,\phi(z),\qquad
# z = \frac{f^\ast-\mu(\theta)}{\sigma(\theta)},
# $$
# where $\Phi$ and $\phi$ are the standard normal CDF and PDF. The first term favors points predicted to be good
# (*exploitation*), and the second favors uncertain points (*exploration*). We maximize EI over random candidates
# spread over the whole box, plus candidates concentrated around the current best point. The latter are needed to
# reach the micrometre-level target.
#
# ### 9.4 What each method uses
#
# | Method | Information per measurement | Prior training | Step limit $\Delta\theta_{\max}$ |
# |---|---|---|---|
# | Response matrix (Section 4) | Both BPM readings (vector) | None, but assumes a linear model | Not applied |
# | Nominal feedback $W_0$ (Section 5.3) | Both BPM readings (vector) | None, but uses the design model $R_0$ | Applied |
# | RL policy (Sections 5–7) | Both BPM readings (vector) | Tens of thousands of simulated steps | Applied |
# | Bayesian optimization | Scalar $f(\theta)$ only | None | Not applied (BO may jump anywhere in the box) |
#
# The comparison is therefore not a contest on equal terms. It shows how prior knowledge (a model, or a policy
# trained in simulation) and the richness of each measurement trade off against the number of measurements needed
# on the machine.

# %%
BO_EPISODES: Final = env_int("BO_EPISODES", 60)
BO_RANDOM_CANDIDATES: Final = 2_000
BO_LOCAL_CANDIDATES: Final = 2_000
BO_LOCAL_SCALE: Final = 0.02
BO_GP_RESTARTS: Final = 1
BO_EI_STD_FLOOR: Final = 1.0e-12
COMPARISON_SEED: Final = SEED + 70_000
RESPONSE_MATRIX_MEASUREMENTS: Final = 5  # four probe readings plus one reading after the correction


@dataclass(frozen=True)
class CorrectionTrace:
    """Max BPM error after each machine measurement of one episode, and whether the target was reached."""

    max_errors: np.ndarray
    is_success: bool

    @property
    def measurements(self) -> int:
        return int(self.max_errors.size)


def orbit_objective(simulator: BeamlineSimulator, kicks: np.ndarray) -> float:
    """Return the squared normalized BPM error for the given kicks."""
    normalized = simulator.bpm_readings(kicks) / BPM_SCALE
    return float(np.dot(normalized, normalized))


def expected_improvement(mean: np.ndarray, std: np.ndarray, best_value: float) -> np.ndarray:
    """Return Expected Improvement for minimization."""
    improvement = best_value - mean
    safe_std = np.maximum(std, BO_EI_STD_FLOOR)
    z_score = improvement / safe_std
    return improvement * scipy.stats.norm.cdf(z_score) + safe_std * scipy.stats.norm.pdf(z_score)


def bayesian_optimization_episode(simulator: BeamlineSimulator, rng: np.random.Generator) -> CorrectionTrace:
    """Minimize the orbit objective for one episode with GP-based Bayesian optimization."""
    kernel = ConstantKernel(1.0, (1.0e-3, 1.0e3)) * RBF(0.5, (1.0e-2, 10.0)) + WhiteKernel(1.0e-6, (1.0e-10, 1.0e-2))
    normalized_points = [np.zeros(2, dtype=np.float64)]
    values = [orbit_objective(simulator, np.zeros(2, dtype=np.float64))]
    max_errors: list[float] = []
    for _ in range(MAX_STEPS):
        gaussian_process = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=BO_GP_RESTARTS,
            random_state=int(rng.integers(2**31 - 1)),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gaussian_process.fit(np.asarray(normalized_points), np.asarray(values))
        best_point = normalized_points[int(np.argmin(values))]
        global_candidates = rng.uniform(-1.0, 1.0, size=(BO_RANDOM_CANDIDATES, 2))
        local_candidates = best_point + BO_LOCAL_SCALE * rng.standard_normal((BO_LOCAL_CANDIDATES, 2))
        candidates = np.clip(np.vstack([global_candidates, local_candidates]), -1.0, 1.0)
        mean, std = gaussian_process.predict(candidates, return_std=True)
        next_point = candidates[int(np.argmax(expected_improvement(mean, std, min(values))))]
        kicks = simulator.lattice.max_kick * next_point
        normalized_points.append(next_point)
        values.append(orbit_objective(simulator, kicks))
        max_errors.append(float(np.max(np.abs(simulator.bpm_readings(kicks)))))
        if max_errors[-1] < TARGET_TOLERANCE:
            return CorrectionTrace(max_errors=np.asarray(max_errors), is_success=True)
    return CorrectionTrace(max_errors=np.asarray(max_errors), is_success=False)


def rl_policy_episode(environment: TwoCorrectorOrbitEnv, policy: LinearPolicy, state: np.ndarray) -> CorrectionTrace:
    """Run a deterministic linear policy from the given reset state and record the error after each step."""
    max_errors: list[float] = []
    terminated = truncated = False
    info = {"is_success": False}
    while not (terminated or truncated):
        state, _, terminated, truncated, info = environment.step(policy.action(state))
        max_errors.append(info["max_abs_reading"])
    return CorrectionTrace(max_errors=np.asarray(max_errors), is_success=bool(info["is_success"]))


def response_matrix_episode(environment: TwoCorrectorOrbitEnv) -> CorrectionTrace:
    """Apply the one-shot response-matrix correction; probe readings are counted as measurements."""
    before, after = response_matrix_correction(environment)
    max_errors = np.full(RESPONSE_MATRIX_MEASUREMENTS, float(np.max(np.abs(before))))
    max_errors[-1] = float(np.max(np.abs(after)))
    return CorrectionTrace(max_errors=max_errors, is_success=bool(max_errors[-1] < TARGET_TOLERANCE))


# %% [markdown]
# ### 9.5 Running the comparison
#
# All methods use environments created with the same seed, so they see identical sequences of hidden
# parameters $(x_0, x'_0, k)$. The number of BO episodes can be changed with the environment variable `BO_EPISODES`;
# each BO episode refits a GP up to 35 times and takes a fraction of a second.

# %%
def run_comparison(
    lattice: Lattice, linear_policies: dict[str, LinearPolicy], episodes: int, seed: int
) -> dict[str, list[CorrectionTrace]]:
    """Run the response matrix, the given linear policies, and BO on identical episodes of the given lattice."""
    bo_rng = np.random.default_rng(seed)
    names = ["Response matrix", *linear_policies, "Bayesian optimization"]
    environments = {name: TwoCorrectorOrbitEnv(lattice, rng_seed=seed) for name in names}
    traces: dict[str, list[CorrectionTrace]] = {name: [] for name in names}
    for _ in range(episodes):
        environments["Response matrix"].reset()
        traces["Response matrix"].append(response_matrix_episode(environments["Response matrix"]))
        for name, linear_policy in linear_policies.items():
            linear_state = environments[name].reset()
            traces[name].append(rl_policy_episode(environments[name], linear_policy, linear_state))
        environments["Bayesian optimization"].reset()
        traces["Bayesian optimization"].append(
            bayesian_optimization_episode(environments["Bayesian optimization"].simulator, bo_rng)
        )
    return traces


def print_comparison(traces: dict[str, list[CorrectionTrace]]) -> None:
    """Print success rate, mean number of measurements, and median final error of each method."""
    for name, method_traces in traces.items():
        successes = np.asarray([trace.is_success for trace in method_traces])
        measurements = np.asarray([trace.measurements for trace in method_traces])
        final_errors_um = 1.0e6 * np.asarray([trace.max_errors[-1] for trace in method_traces])
        mean_successful = float(np.mean(measurements[successes])) if np.any(successes) else float("nan")
        print(
            f"  {name:40s}:",
            f"success={100.0 * np.mean(successes):5.1f}%,",
            f"mean measurements (successful)={mean_successful:5.2f},",
            f"median final error={np.median(final_errors_um):9.2f} um",
        )


comparison_traces = run_comparison(
    LATTICE, {"Nominal feedback W0": nominal_policy, "RL policy (DDPG)": policy}, BO_EPISODES, COMPARISON_SEED
)
print(f"Comparison over {BO_EPISODES} identical episodes:")
print_comparison(comparison_traces)

# %% [markdown]
# ### 9.6 Convergence and measurement count
#
# The left panel shows the median over episodes of the best $\max\lvert x_{\mathrm{BPM}}\rvert$ reached so far
# versus the number of machine measurements (the bands span the 25th to 75th percentiles). The RL policy and BO
# continue only until success, so a finished episode keeps its final value. The response-matrix method
# reaches the target after its fifth measurement; its residual (about $10^{-10}$ µm, numerical round-off) lies below
# the plotted range and is drawn at the lower edge of the panel. The right panel shows the distribution of the number
# of measurements needed for success.

# %%
def best_so_far_matrix(traces: list[CorrectionTrace], length: int) -> np.ndarray:
    """Return best-so-far max BPM error per measurement, padded with the final value after termination."""
    matrix = np.empty((len(traces), length), dtype=np.float64)
    for row, trace in enumerate(traces):
        running_best = np.minimum.accumulate(trace.max_errors)
        matrix[row, : running_best.size] = running_best[:length]
        matrix[row, running_best.size :] = running_best[-1]
    return matrix


QUARTILES: Final = (25.0, 50.0, 75.0)
CONVERGENCE_PLOT_FLOOR_UM: Final = 0.1


def plot_comparison(traces: dict[str, list[CorrectionTrace]], title_suffix: str = "") -> None:
    """Plot the best-so-far error versus measurements and the histogram of measurements to success."""
    measurement_axis = np.arange(1, MAX_STEPS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for name, method_traces in traces.items():
        best_so_far_um = np.maximum(1.0e6 * best_so_far_matrix(method_traces, MAX_STEPS), CONVERGENCE_PLOT_FLOOR_UM)
        lower, median, upper = np.percentile(best_so_far_um, QUARTILES, axis=0)
        line = axes[0].plot(measurement_axis, median, marker=".", label=name)[0]
        axes[0].fill_between(measurement_axis, lower, upper, color=line.get_color(), alpha=0.2)
    axes[0].axhline(1.0e6 * TARGET_TOLERANCE, color="black", linestyle="--", linewidth=1.0, label="Success threshold")
    axes[0].set_yscale("log")
    axes[0].set_ylim(bottom=0.5 * CONVERGENCE_PLOT_FLOOR_UM)
    axes[0].set(
        xlabel="Machine measurements",
        ylabel="Best max BPM error so far [um]",
        title=f"Convergence (median and interquartile range){title_suffix}",
    )
    axes[0].legend(fontsize=8)

    measurement_bins = np.arange(0.5, MAX_STEPS + 1.5, 1.0)
    largest_successful_count = 1
    for name, method_traces in traces.items():
        successful_counts = [trace.measurements for trace in method_traces if trace.is_success]
        axes[1].hist(successful_counts, bins=measurement_bins, alpha=0.6, label=name)
        largest_successful_count = max([largest_successful_count, *successful_counts])
    axes[1].set_xlim(0.5, largest_successful_count + 1.5)
    axes[1].set(xlabel="Measurements to success", ylabel="Episodes", title=f"Measurements needed for success{title_suffix}")
    axes[1].legend(fontsize=8)
    fig.tight_layout()


plot_comparison(comparison_traces)

# %% [markdown]
# ### 9.7 Discussion
#
# - **Response matrix:** with a linear, noiseless model and vector BPM readings, a few probe measurements determine
#   the correction exactly, down to numerical round-off. This is the most accurate method whenever its model
#   assumptions hold.
# - **Nominal feedback $W_0$ and RL policy:** both correct a new episode without probing, but only to the 10 µm
#   target, and both are limited to small kick increments, which mainly sets their number of steps.
#   The RL policy's correction strategy is *amortized*: the cost was paid in advance by tens of thousands of
#   simulated training steps. The nominal feedback reaches a similar or better result with no training at all,
#   because the design model is good enough for this machine (Section 8).
# - **Bayesian optimization:** BO needs no model and no training, and it sees only a scalar objective. It must
#   discover the two-dimensional structure of $f(\theta)$ from scratch in every episode, so it needs many more
#   measurements than the other methods, and some episodes may not reach the micrometre-level target within
#   the budget. In return, it extends naturally to nonlinear objectives, noisy measurements, and
#   knobs for which no model or simulator exists.
#
# In practice these methods are complementary. For example, a policy or model provides a good starting point, and
# BO fine-tunes it on the real machine. Here the BO objective is an analytic quadratic bowl; on a real machine the GP
# noise term $\sigma_n$ becomes essential.
#
# **Exercises for this section.**
#
# 1. Replace Expected Improvement by the upper (lower) confidence bound $\mu(\theta)-\kappa\sigma(\theta)$ and vary $\kappa$.
# 2. Add BPM noise and observe how the fitted `WhiteKernel` level and the BO convergence change.
# 3. Give BO the same step limit as RL by restricting candidates to $\lvert\theta-\theta_{\mathrm{last}}\rvert\le\Delta\theta_{\max}$.
# 4. Start BO from the kicks proposed by one RL step instead of $\theta=0$ (warm-started BO).

# %% [markdown]
# ### 9.8 Comparison for the strong quadrupole
#
# The same comparison for the strong-quadrupole scenario of Section 7.5 ($k \in [5, 7]$ m$^{-1}$ with three times larger
# kick limits). The linear policies are the feedback from the wrong model ($k_0 = 0$), the feedback from the correct
# model ($k_\mathrm{ref} = 6$ $m^{-1}$), and the best checkpoints of the two RL runs with the training seed `SEED`
# (Section 7.5). The response-matrix method measures the response in every episode, so it does not depend on a model.
# BO searches the larger kick box of $\pm 12$ mrad.
#
# Things to look for:
#
# - The response-matrix method does not use a model of the optics: it still needs exactly five measurements and
#   corrects the orbit down to numerical round-off.
# - BO does not use a model either, but it becomes much harder: the kick box is three times larger, and the objective
#   is a narrow, elongated valley because the response matrix is nearly singular. Many episodes do not reach the
#   target within the 35 measurements.
# - The feedback from the wrong model fails in every episode (the orbit error grows until the kicks hit their limits),
#   while RL works without any model. With prior knowledge from the correct model, the fixed feedback needs fewer
#   measurements than RL, but more than the response-matrix method, because it is exact only at $k = k_\mathrm{ref}$.

# %%
strong_comparison_traces = run_comparison(
    STRONG_LATTICE,
    {
        "W0 from the wrong model (k0 = 0)": wrong_policy,
        f"W0 from the correct model (k_ref = {STRONG_K_REF})": correct_policy,
        "RL from W = 0": strong_run_for_seed("zero").best_policy,
        "RL from the wrong W0": strong_run_for_seed("nominal").best_policy,
    },
    BO_EPISODES,
    COMPARISON_SEED + 1,
)
print(f"Strong quadrupole, comparison over {BO_EPISODES} identical episodes:")
print_comparison(strong_comparison_traces)
plot_comparison(strong_comparison_traces, title_suffix=", strong k")
