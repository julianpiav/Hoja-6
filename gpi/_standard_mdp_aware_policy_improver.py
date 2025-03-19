from gpi._base import GeneralPolicyIterationComponent


class StandardMDPAwarePolicyImprover(GeneralPolicyIterationComponent):

    def __init__(self, mdp):
        self.mdp = mdp

    def step(self):
        """
        Actualiza la política en el workspace basándose en los valores de acción-estado.
        Implementa el paso de mejora de política del algoritmo de Policy Iteration.
        """
        # Si no existen los valores Q, no se puede mejorar la política
        if self.workspace.q is None:
            return

        # Recorrer todos los estados definidos en el MDP
        for s in self.mdp.states:
            # Si el estado es terminal, se ignora pues no hay acciones aplicables
            if self.mdp.is_terminal_state(s):
                continue

            # Se asume que en el workspace, self.workspace.q es un diccionario
            # donde cada estado 's' tiene asociado otro diccionario con pares {acción: Q(s,a)}
            q_values = self.workspace.q[s]

            # Se selecciona la acción con el valor Q máximo en el estado s.
            # La función max() aplicada al diccionario usa, con key=q_values.get,
            # el valor de cada clave para la comparación.
            best_action = max(q_values, key=q_values.get)

            # Se actualiza la política en el workspace para el estado s con la mejor acción encontrada.
            self.workspace.policy[s] = best_action