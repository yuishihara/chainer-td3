import chainer
import chainer.links as L
import chainer.functions as F


class TD3Critic(chainer.Chain):
    def __init__(self, state_dim, action_num):
        super(TD3Critic, self).__init__()
        with self.init_scope():
            self._linear1 = L.Linear(
                in_size=(state_dim+action_num), out_size=400)
            self._linear2 = L.Linear(in_size=400, out_size=300)
            self._linear3 = L.Linear(in_size=300, out_size=1)
        self._state_dim = state_dim
        self._action_num = action_num

    def __call__(self, s):
        h = self._linear1(s)
        h = F.relu(h)
        h = self._linear2(h)
        h = F.relu(h)
        q = self._linear3(h)
        return q


if __name__ == "__main__":
    pass
