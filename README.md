# Reinforcement Learning Projects

A collection of my **Reinforcement Learning coursework and experiments**, focused on understanding how an agent can make decisions in an environment using rewards, value functions, policies, and sequential decision-making.

I built these tasks to understand the algorithms from the inside instead of treating reinforcement learning as a black box.

## What This Repository Covers

- **Grid-World MDP** — modelling states, actions, rewards, and stochastic transitions.
- **Value Iteration** — finding optimal state values using the Bellman optimality equation.
- **Policy Iteration** — evaluating and improving policies until they converge.
- **Vectorized Dynamic Programming** — using NumPy operations to avoid unnecessary state-by-state computation.
- **Soft Actor-Critic (SAC)** — exploring an off-policy actor-critic method for continuous control with the Pendulum environment.
- **A3C** — studying asynchronous advantage actor-critic ideas with CartPole.

## The Main Idea

The simplest way I think about RL is:

> **Take an action → observe what happened → receive a reward → improve the decision.**

The coursework helped me connect that idea to actual algorithms such as Value Iteration, Policy Iteration, SAC, and A3C.

## Repository Structure

```text
Reinforcement-Learning-projects/
├── gridworld.py
├── assets/
├── EXP-7_SAC_Pendulum.ipynb
├── EXP-8_A3C_CartPole.ipynb
└── README.md
```

## Environment: Icy Grid World

The core Grid-World experiment uses a **10×10 stochastic environment** where the robot needs to reach a goal while avoiding a trap. Actions are probabilistic, so the agent has to account for the possibility of slipping into another direction.

The implementation compares Value Iteration and Policy Iteration and visualizes the resulting value map and policy.

## Results

The current Grid-World implementation reports:

- Value Iteration: **61 iterations** to converge
- Policy Iteration: **4 policy-improvement steps**
- Start-state value: **6.195**
- Policy agreement: **97 / 100 states**, with the remaining differences caused by value ties

## Tech Stack

**Python · NumPy · SciPy · Matplotlib · Reinforcement Learning · Markov Decision Processes · Dynamic Programming · Actor-Critic Methods**

## Running the Projects

For the Grid-World implementation:

```bash
pip install numpy scipy matplotlib
python gridworld.py
```

The reinforcement-learning notebooks can be opened in **Jupyter Notebook, JupyterLab, or Google Colab**. Their required libraries are listed inside the notebooks.

## What I Learned

The biggest takeaway from these tasks was that RL is not simply about training an agent until it works. The interesting part is understanding **why a policy changes, how rewards propagate through states, and how uncertainty affects decisions**.

These experiments form part of my college coursework and my foundation in reinforcement learning.

## Coursework Submission

Submission link: https://forms.gle/z5cx2JNkG5PHUaoR7

---

Learning reinforcement learning by actually building the algorithms — not just reading about them.
