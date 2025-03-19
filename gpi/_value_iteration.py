from _base import GeneralPolicyIterationComponent


class StandardMDPAwareValueIteration(GeneralPolicyIterationComponent):

    def __init__(self, mdp):
        self.mdp = mdp

    @property
    def upper_bound_for_state_value_estimation_error(self):
        """
        Calcula el límite superior para el error de estimación de valor de estado
        en Value Iteration.
        """
        if not hasattr(self, 'max_delta'):
            return float('inf')
        
        gamma = self.workspace.gamma
        return self.max_delta * gamma / (1 - gamma) if gamma < 1 else float('inf')

def step(self):
    """
    Actualiza la política y los valores de estado en una sola pasada.
    Implementa una iteración del algoritmo de Value Iteration.
    """
    # Inicializar valores si es necesario
    if self.workspace.v is None:
        self.workspace.replace_v({s: 0.0 for s in self.mdp.states})
    
    if self.workspace.q is None:
        self.workspace.replace_q({s: {a: 0.0 for a in self.mdp.get_actions_in_state(s)} 
                                for s in self.mdp.states if not self.mdp.is_terminal_state(s)})
    
    # Guardar valores anteriores para calcular el error
    old_v = self.workspace.v.copy()
    self.max_delta = 0
    
    # Actualizar valores de acción-estado (q-values)
    for s in self.mdp.states:
        if self.mdp.is_terminal_state(s):
            self.workspace.v[s] = self.mdp.get_reward(s)
            continue
        
        # Calcular nuevos q-values para cada acción
        for a in self.mdp.get_actions_in_state(s):
            next_state_dist = self.mdp.get_transition_distribution(s, a)
            q_value = self.mdp.get_reward(s) + self.workspace.gamma * sum(
                prob * old_v[next_s] for next_s, prob in next_state_dist.items()
            )
            self.workspace.q[s][a] = q_value
        
        # Encontrar el mejor valor de estado según las acciones disponibles
        best_action = max(self.mdp.get_actions_in_state(s), key=lambda a: self.workspace.q[s][a])
        best_value = self.workspace.q[s][best_action]
        
        # Actualizar el valor de estado
        self.workspace.v[s] = best_value
        
        # Actualizar el error máximo
        self.max_delta = max(self.max_delta, abs(old_v[s] - best_value))
    
    # Actualizar la política para elegir la mejor acción según los q-values
    def policy(s):
        if self.mdp.is_terminal_state(s):
            return None
        actions = self.mdp.get_actions_in_state(s)
        if not actions:
            return None
        return max(actions, key=lambda a: self.workspace.q[s][a])
    
    self.workspace.replace_policy(policy)