# State-Value-Function-Estimation-using-Monte-Carlo-Prediction
### Thiyagarajan A - 212222240110
## AIM
To compare the performance of Monte Carlo Control (model-free reinforcement learning) and Value Iteration (model-based dynamic programming) in estimating the optimal state-value function V(s) for the FrozenLake-v1 environment using Gym.

## ALGORITHM
### Monte Carlo Control (First Visit)

1. Initialize Q-table arbitrarily.
2. Generate trajectory using epsilon-greedy policy.
3. Calculate returns using discounted rewards.
4. Update Q-values using learning rate (alpha).
5. Derive policy from Q-values.

### Value Iteration (Dynamic Programming)

1. Initialize value function V(s) arbitrarily.
2. Iteratively update V(s) using Bellman optimality equation:
   V(s) = max_a Σ [P(s'|s,a) * (R + γ * V(s'))]
3. Stop when value updates are below a small threshold (θ).
4. Derive policy by choosing best action for each state.

## CODE

```python
# --- IMPORTS ---
import gym, gym_walk
import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import count
from tqdm import tqdm

# --- SETTINGS ---
random.seed(123); np.random.seed(123)
env = gym.make('FrozenLake-v1', is_slippery=True)
P = env.env.P

# --- PRINT FUNCTION ---
def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

# --- DECAY SCHEDULE FUNCTION ---
def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start = -2, log_base=10):
    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps
    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    return np.pad(values, (0, rem_steps), 'edge')

# --- GENERATE TRAJECTORY ---
def generate_trajectory(select_action, Q, epsilon, env, max_steps=200):
    done, trajectory = False, []
    while not done:
        state = env.reset()
        for t in count():
            action = select_action(state, Q, epsilon)
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, action, reward, next_state, done))
            if done or t >= max_steps - 1:
                break
            state = next_state
    return np.array(trajectory, object)

# --- MONTE CARLO CONTROL ---
def mc_control(env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
               init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9,
               n_episodes=3000, max_steps=200, first_visit=True):

    nS, nA = env.observation_space.n, env.action_space.n
    discounts = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    Q = np.zeros((nS, nA), dtype=np.float64)

    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))

    for e in tqdm(range(n_episodes), leave=False):
        trajectory = generate_trajectory(select_action, Q, epsilons[e], env, max_steps)
        visited = np.zeros((nS, nA), dtype=bool)

        for t, (state, action, reward, _, _) in enumerate(trajectory):
            if visited[state][action] and first_visit:
                continue
            visited[state][action] = True
            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
            Q[state][action] += alphas[e] * (G - Q[state][action])

    V = np.max(Q, axis=1)
    pi = np.argmax(Q, axis=1)
    return Q, V, pi

# --- VALUE ITERATION ---
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break
        V = np.max(Q, axis=1)
    pi = np.argmax(Q, axis=1)
    return V, pi

```

## OUTPUT
![image](https://github.com/user-attachments/assets/a7ca3af4-654b-4460-9443-b809fdc3cd52)
![image](https://github.com/user-attachments/assets/e81d8860-60f4-4ffe-a944-9fc87aeea044)

## RESULT

Dynamic Programming outperformed Monte Carlo by providing more accurate and stable state-value estimates in the FrozenLake environment.
