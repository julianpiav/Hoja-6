import numpy as np

class Analyzer:

    def __init__(self, mdp):
        self.mdp = mdp
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
            It uses a policy evaluator that solves the set of linear equations in the MDP in order to compute the statee values.
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
            true_q = self.true_q[s]
            estimated_q = history[t][1][s]
            ax.scatter(true_q,estimated_q, label=str(approach))

            ax.plot([min(self.true_q[s]), max(self.true_q[s])],
                [min(self.true_q[s]), max(self.true_q[s])], 'k--')
            ax.set_xlabel("True Q-values")
            ax.legend()
            plt.show()


    def plot_rmse_over_time(self, ax=None):
        """
            Creates a plot with one curve for each approach, with the time on the x-axis.
            For the values on the y-axis, it compares, for each state, the true state value with the estimated value by each approach, and computes the squared error. Averaging over these values and taking the square root yields the RMSE, which is to be shown on the y-axis.
        """
        if ax is None:
            fig, ax = plt.subplots()

        for apprach, history in self._history.items():
            rmse_values = []
            for t in range(len(history)):
                estimated_v = np.array([history[t][0][s] for s in self.states])
                rmse = np.sqrt(np.mean((estimated_v - self.true_v) ** 2))
                rmse_values.append(rmse)

            ax.plot(range(len(history)), rmse_values, label=str(apprach))

        ax.set_xlabel("Time Steps")
        ax.set_ylabel("RMSE")
        ax.legend()
        plt.show()