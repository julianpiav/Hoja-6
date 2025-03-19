from _base import GeneralPolicyIterationComponent


class StandardMDPAwarePolicyImprover(GeneralPolicyIterationComponent):

    def __init__(self, mdp):
        self.mdp = mdp
    
    def step(self):
        """
        Actualiza la política en el workspace basándose en los valores de acción-estado.
        Implementa el paso de mejora de política del algoritmo de Policy Iteration.
        """
    # Si no hay valores de acción-estado, no podemos mejorar la política
        if self.workspace.q is None:
            return
    
    # Inicializar la política si es necesario
        if self.workspace.policy is None:
            # Crear una función que represente la política
            def policy(s):
                if self.mdp.is_terminal_state(s):
                    return None
                # Elegir la acción con el mayor valor de acción-estado
                actions = self.mdp.get_actions_in_state(s)
                if not actions:
                    return None
                return max(actions, key=lambda a: self.workspace.q[s][a])
            
            self.workspace.replace_policy(policy)
            return
        
        # Actualizar la política para elegir la acción con el mayor valor de acción-estado
        old_policy = self.workspace.policy
        
        def new_policy(s):
            if self.mdp.is_terminal_state(s):
                return None
            actions = self.mdp.get_actions_in_state(s)
            if not actions:
                return None
            return max(actions, key=lambda a: self.workspace.q[s][a])
        
        self.workspace.replace_policy(new_policy)