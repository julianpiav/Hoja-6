import numpy as np

from abc import ABC


class MDP(ABC):
    """
        Abstract class to be used as a common interface by algorithms based on MDPs
    """

    @property
    def init_states(self) -> list:
        """

        :return: list of states in which the agent might start (uniformly distributed)
        """
        raise NotImplementedError

    @property
    def states(self) -> list:
        """

        :return: list of all possible states of the MDP (might not be implemented if this set is too large or infinite).
        """
        raise NotImplementedError

    def is_terminal_state(self, s) -> bool:
        """

        :param s: state to be checked for being terminal
        :return: True iff `s` is terminal (False otherwise)
        """
        return len(self.get_actions_in_state(s)) == 0

    @property
    def actions(self) -> list:
        """

        :return: list of all actions the agent could ever execute in any state.
        """
        actions = set()
        for s in self.get_states():
            actions |= set(self.get_actions_in_state(s))
        return list(actions)

    def get_actions_in_state(self, s) -> list:
        """

        :param s: state for which the set of applicable actions is queried
        :return: list of actions applicable for the agent in state `s`
        """
        raise NotImplementedError

    def get_reward(self, s) -> float:
        """

        :param s: state `s` for which the reward is being queried
        :return:  reward (float) the agent receives when entering state `s`
        """
        raise NotImplementedError

    def get_transition_distribution(self, s, a) -> dict:
        """

        :param s: state from which the agent departs
        :param a: action executed by the agent
        :return: dictionary describing the distribution among states in which the agent will end up
        """
        raise NotImplementedError







