from _base import GeneralPolicyIterationComponent


class IncrementalMDPAwarePolicyEvaluator(GeneralPolicyIterationComponent):

    def __init__(self, mdp):
        self.mdp = mdp
    
    @property
    def upper_bound_for_state_value_estimation_error(self):
        """
        Calcula el límite superior para el error de estimación de valor de estado.
        Basado en la teoría de convergencia en Policy Evaluation.
        """
    # El límite de error para evaluación de políticas es γ/(1-γ) veces el cambio máximo
        if self.workspace.v is None:
            return float('inf')
    
        gamma = self.workspace.gamma
        return gamma / (1 - gamma) * self.max_delta if gamma < 1 else float('inf')

def step(self):
    """
    Actualiza los valores de estado y acción-estado basados en la política actual.
    Implementa una iteración de Bellman para evaluación de políticas.
    """
    # Si no hay política o valores inicializados, inicialízalos
    if self.workspace.policy is None:
        return
    
    if self.workspace.v is None:
        # Inicializar valores de estado a cero
        self.workspace.replace_v({s: 0.0 for s in self.mdp.states})
    
    # Inicializar valores de acción-estado si es necesario
    if self.workspace.q is None:
        self.workspace.replace_q({s: {a: 0.0 for a in self.mdp.get_actions_in_state(s)} 
                                for s in self.mdp.states if not self.mdp.is_terminal_state(s)})
    
    # Actualizar valores de estado usando ecuación de Bellman para evaluación de políticas
    policy = self.workspace.policy
    old_v = self.workspace.v.copy()
    self.max_delta = 0
    
    # Para cada estado, actualizar su valor según la política actual
    for s in self.mdp.states:
        if self.mdp.is_terminal_state(s):
            self.workspace.v[s] = self.mdp.get_reward(s)
            continue
        
        a = policy(s)
        # Calcular el nuevo valor de estado usando ecuación de Bellman
        next_state_dist = self.mdp.get_transition_distribution(s, a)
        new_value = self.mdp.get_reward(s) + self.workspace.gamma * sum(
            prob * self.workspace.v[next_s] for next_s, prob in next_state_dist.items()
        )
        
        # Actualizar el valor de estado
        self.workspace.v[s] = new_value
        
        # Actualizar el error máximo
        self.max_delta = max(self.max_delta, abs(old_v[s] - new_value))
    
    # Actualizar valores de acción-estado (q-values) basados en los valores de estado actualizados
    for s in self.mdp.states:
        if self.mdp.is_terminal_state(s):
            continue
        
        for a in self.mdp.get_actions_in_state(s):
            # Calcular el valor de acción-estado usando ecuación de Bellman
            next_state_dist = self.mdp.get_transition_distribution(s, a)
            q_value = self.mdp.get_reward(s) + self.workspace.gamma * sum(
                prob * self.workspace.v[next_s] for next_s, prob in next_state_dist.items()
            )
            self.workspace.q[s][a] = q_value