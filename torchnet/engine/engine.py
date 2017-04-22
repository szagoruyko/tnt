import collections
from .hook import Hook, HooksList, StopEngine


class Engine(object):
    def __init__(self, hook=None):
        if isinstance(hook, collections.Sequence):
            self.hook = HooksList(hook)
        else:
            self.hook = hook or Hook()

    def train(self, network, iterator, maxepoch, optimizer):
        try:
            state = {'network': network,
                     'iterator': iterator,
                     'maxepoch': maxepoch,
                     'optimizer': optimizer,
                     'epoch': 0,
                     't': 0,
                     'train': True}

            # start training
            self.hook.on_start(state)

            # start running epochs
            while state['epoch'] < state['maxepoch']:
                self.hook.on_start_epoch(state)

                for sample in state['iterator']:
                    state['sample'] = sample
                    self.hook.on_sample(state)

                    def closure():
                        # forward
                        loss, output = state['network'](state['sample'])
                        state['output'] = output
                        state['loss'] = loss
                        self.hook.on_forward(state)

                        # for backward
                        loss.backward()

                        # to free memory in save_for_backward
                        state['output'] = None
                        state['loss'] = None

                        return loss

                    state['optimizer'].zero_grad()
                    state['optimizer'].step(closure)
                    state['t'] += 1

                self.hook.on_end_epoch(state)
                state['epoch'] += 1

        except StopEngine:
            pass

        # training complete
        self.hook.on_end(state)

        return state

    def test(self, network, iterator):
        try:
            state = {
                'network': network,
                'iterator': iterator,
                't': 0,
                'train': False}

            # start testing
            self.hook.on_start(state)
            for sample in state['iterator']:
                state['sample'] = sample
                self.hook.on_sample(state)

                def closure():
                    # forward
                    loss, output = state['network'](state['sample'])
                    state['output'] = output
                    state['loss'] = loss
                    self.hook.on_forward(state)

                    # to free memory in save_for_backward
                    state['output'] = None
                    state['loss'] = None

                    return loss

                state['optimizer'].zero_grad()
                state['optimizer'].step(closure)
                self.hook.on_update(state)
                closure()
                state['t'] += 1

        except StopEngine:
            pass

        # testing complete
        self.hook.on_end(state)
        return state
