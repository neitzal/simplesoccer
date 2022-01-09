from simplesoccer.mini_env_states import SoccerStates


class EvalPolicy:
    def compute_actions(self, states: SoccerStates):
        raise NotImplementedError()
