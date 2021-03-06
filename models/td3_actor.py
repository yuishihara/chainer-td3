import chainer
import chainer.links as L
import chainer.functions as F


class TD3Actor(chainer.Chain):
    def __init__(self, state_dim, action_num):
        super(TD3Actor, self).__init__()
        with self.init_scope():
            self._linear1 = L.Linear(in_size=state_dim, out_size=400)
            self._linear2 = L.Linear(in_size=400, out_size=300)
            self._linear3 = L.Linear(in_size=300, out_size=action_num)
        self._action_num = action_num

    def __call__(self, s):
        h = self._linear1(s)
        h = F.relu(h)
        h = self._linear2(h)
        h = F.relu(h)
        h = self._linear3(h)
        action = F.tanh(h)
        return action


if __name__ == "__main__":
    pass
