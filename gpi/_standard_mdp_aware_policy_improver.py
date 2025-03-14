from _base import GeneralPolicyIterationComponent


class StandardMDPAwarePolicyImprover(GeneralPolicyIterationComponent):

    def __init__(self, mdp):
        self.mdp = mdp
    
    def step(self):
        pass  # your code here to update the policy in the workspace based on the state or state-action values in the workspace