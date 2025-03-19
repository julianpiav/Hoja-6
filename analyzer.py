from math import gamma

import numpy as np
from matplotlib import pyplot as plt

from gpi._base import GeneralPolicyIteration


class Analyzer:

    def __init__(self, mdp, workspace):
        self.mdp = mdp
        self.workspace = workspace
        self._history = {}
    
    @property
    def history(self):
        return self._history
    
    def prepare(self):
        """
            This function computes the optimal policy and the true state and state-action values for all states in the MDP.
        """
        self.true_v = self.solve_linear_system()
        self.true_q = self.compute_q_values(self.true_v)

    def reset(self, approaches: list):
        """
            :param approaches: a list of algorithm objects, each of which with a function `step` and properties `v`, `q`, and `policy`
            :return: Nothing

            Tells the object about which approaches are going to be analyzed. Resets the history
        """
        self._history = {approach: [] for approach in approaches}
        self.approaches = approaches

    def step(self):
        """
            Executes a single step of all analyzed approaches and then observes their estimates for state values and state-action values.
            It keeps track of the estimates for state and state-action values produced by each approach over time (added to the history)
        """
        for approach in self.approaches:
            approach.step()
            self._history[approach].append((approach.v.copy(), approach.q.copy()))
    
    def get_true_q_values_for_policy(self, policy):
        """
            :param policy: The policy for which true q-values are to be computed
            :returns: The true q-values of policy `policy`

            Determines the true q-values for the given policy.
            It uses a policy evaluator that solves the set of linear equations in the MDP in order to compute the state values.
            Then it computes the q-values from the state-values.
        """
        v_pi = self.solve_linear_system(policy)
        return self.compute_q_values(v_pi)

    def plot_q_value_comparison(self, s, t, ax=None):
        """
            Creates a scatter plot where the q-values of state `s` are shown (true on x-axis and estimated values on y-axis) after `t` steps have elapsed.
            Different colors are used for different analyzed approaches.
            The diagonal black dashed line shows where predictions and true values would coincide
        """
        if ax is None:
            fig, ax = plt.subplots()

        for approach, history in self._history.items():
            true_q = np.array(self.true_q[s])
            estimated_q = np.array(history[t][1][s])
            if true_q.shape == estimated_q.shape:
                ax.scatter(true_q, estimated_q, label=str(approach))

        ax.plot([min(self.true_q[s]), max(self.true_q[s])],
                [min(self.true_q[s]), max(self.true_q[s])], 'k--')
        ax.set_xlabel("True Q-values")
        ax.legend()
        if ax is None:
            plt.show()

    def plot_rmse_over_time(self, ax=None):
        """
            Creates a plot with one curve for each approach, with the time on the x-axis.
            For the values on the y-axis, it compares, for each state, the true state value with the estimated value by each approach, and computes the squared error. Averaging over these values and taking the square root yields the RMSE, which is to be shown on the y-axis.
        """
        if ax is None:
            fig, ax = plt.subplots()

        for approach, history in self._history.items():
            rmse_values = []
            for t in range(len(history)):
                estimated_v = np.array([history[t][0][s] for s in self.mdp.states])
                if len(estimated_v) > 0:
                    rmse = np.sqrt(np.mean((estimated_v - self.true_v) ** 2))
                    rmse_values.append(rmse)

            ax.plot(range(len(history)), rmse_values, label=str(approach))

        ax.set_xlabel("Time Steps")
        ax.set_ylabel("RMSE")
        ax.legend()
        plt.show()

    def run_gpi_variants(self):
        """
        Ejecuta GPI con 1, 10 y 100 iteraciones antes de la mejora de política.
        Grafica el RMSE y diferencias en Q-values.
        """
        gpi_variants = [1, 10, 100]
        results = {}
        q_histories = {}

        for updates in gpi_variants:
            gpi_instance = GeneralPolicyIteration(self.mdp, updates)
            self.reset([gpi_instance])
            history = []
            q_history = []

            for _ in range(50):
                self.step()
                history.append(self.compute_rmse(gpi_instance.workspace.v))
                q_history.append(gpi_instance.workspace.q.copy())

            results[updates] = history
            q_histories[updates] = q_history

        self.plot_rmse_over_time(results)
        self.plot_q_value_differences(q_histories)

    def plot_q_value_differences(self, histories):
        """
        Crea 10 gráficos 4x4 donde cada celda representa un estado,
        mostrando diferencias en los valores Q(s, a) entre las 3 variantes de GPI.
        """
        iterations_to_plot = min(10, len(histories[1]))

        fig, axes = plt.subplots(10, 1, figsize=(8, 40))

        for t in range(iterations_to_plot):
            diffs = np.zeros((4, 4))

            for s in self.mdp.states:
                row, col = divmod(s, 4)
                q_1 = histories[1][t][s] if s in histories[1][t] else {}
                q_10 = histories[10][t][s] if s in histories[10][t] else {}
                q_100 = histories[100][t][s] if s in histories[100][t] else {}

                best_1 = max(q_1, key=q_1.get, default=None)
                best_10 = max(q_10, key=q_10.get, default=None)
                best_100 = max(q_100, key=q_100.get, default=None)

                diffs[row, col] = int((best_1 != best_10) or (best_10 != best_100))

            ax = axes[t]
            ax.imshow(diffs, cmap="coolwarm", interpolation="nearest")
            ax.set_title(f"Iteración {t + 1}")
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.show()

    def compute_rmse(self, estimated_v):
        """
        Calcula el RMSE entre los valores de estado estimados y los valores verdaderos.
        """
        return np.sqrt(np.mean((np.array(estimated_v) - np.array(self.true_v)) ** 2))

    def solve_linear_system(self):
        """
        Resuelve un sistema de ecuaciones para calcular los valores óptimos de estado.
        """
        num_states = len(self.mdp.states)
        A = np.eye(num_states)  # Matriz identidad
        b = np.zeros(num_states)  # Vector de recompensas

        for i, s in enumerate(self.mdp.states):
            if self.mdp.is_terminal_state(s):
                A[i, i] = 1
                b[i] = self.mdp.get_reward(s)
            else:
                for a in self.mdp.get_actions_in_state(s):
                    transitions = self.mdp.get_transition_distribution(s, a)
                    for s_prime, prob in transitions.items():
                        if s_prime not in self.mdp.states:
                            print(f" Advertencia: Estado {s_prime} no es válido, ignorando.")
                            continue  # Ignorar estados no válidos
                        A[i, self.mdp.states.index(s_prime)] -= self.workspace.gamma * prob
                    b[i] += self.mdp.get_reward(s)
        print("Matriz A:\n", A)
        print("Vector b:\n", b)
        return np.linalg.pinv(A) @ b

    def compute_q_values(self, v):
        """
        Calcula los valores Q(s, a) basados en los valores de estado V(s).
        """
        q_values = {s: {} for s in self.mdp.states}

        for s in self.mdp.states:
            if self.mdp.is_terminal_state(s):
                continue

            for a in self.mdp.get_actions_in_state(s):
                transitions = self.mdp.get_transition_distribution(s, a)
                q_values[s][a] = sum(
                    prob * (self.mdp.get_reward(s) + self.workspace.gamma * v[s_prime])
                    for s_prime, prob in transitions.items()
                )

        return q_values
