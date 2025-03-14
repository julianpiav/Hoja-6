from _base import GeneralPolicyIterationComponent


class StandardMDPAwareValueIteration(GeneralPolicyIterationComponent):

    def __init__(self, mdp):
        self.mdp = mdp

    @property
    def upper_bound_for_state_value_estimation_error(self):
        pass  # your code here to compute the upper bound for the estimation error to the true optimal policy state values v
    
    def step(self):
        pass  # your code here to update both the policy and the state values (and state-action values) in the workspace
