from _base import GeneralPolicyIterationComponent


class IncrementalMDPAwarePolicyEvaluator(GeneralPolicyIterationComponent):

    def __init__(self, mdp):
        self.mdp = mdp
    
    @property
    def upper_bound_for_state_value_estimation_error(self):
        pass  # your code here to compute the upper bound for the estimation error to v^\pi

    def step(self):
        pass  # your code here to update the state values and state-action values in the workspace based on the policy in the workspace