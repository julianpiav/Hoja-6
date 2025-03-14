class Analyzer:

    def __init__(self, mdp):
        self.mdp = mdp
        self._history = None
    
    @property
    def history(self):
        pass
    
    def prepare(self):
        """
            This function computes the optimal policy and the true state and state-action values for all states in the MDP.
        """
        pass

    def reset(self, approaches: list):
        """
            :param approaches: a list of algorithm objects, each of which with a function `step` and properties `v`, `q`, and `policy`
            :return: Nothing

            Tells the object about which approaches are going to be analyzed. Resets the history
        """
        pass

    def step(self):
        """
            Executes a single step of all analyzed approaches and then observes their estimates for state values and state-action values.
            It keeps track of the estimates for state and state-action values produced by each approach over time (added to the history)
        """
    
    def get_true_q_values_for_policy(self, policy):
        """
            :param policy: The policy for which true q-values are to be computed
            :returns: The true q-values of policy `policy`

            Determines the true q-values for the given policy.
            It uses a policy evaluator that solves the set of linear equations in the MDP in order to compute the statee values.
            Then it computes the q-values from the state-values.
        """
        pass
    
    def plot_q_value_comparison(self, s, t, ax=None):
        """
            Creates a scatter plot where the q-values of state `s` are shown (true on x-axis and estimated values on y-axis) after `t` steps have elapsed.
            Different colors are used for different analyzed approaches.
            The diagonal black dashed line shows where predictions and true values would coincide
        """
        pass

    def plot_rmse_over_time(self, ax=None):
        """
            Creates a plot with one curve for each approach, with the time on the x-axis.
            For the values on the y-axis, it compares, for each state, the true state value with the estimated value by each approach, and computes the squared error. Averaging over these values and taking the square root yields the RMSE, which is to be shown on the y-axis.
        """
        pass
