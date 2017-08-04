from tqdm import tqdm
import torch
import torch.optim
import torchnet as tnt
from torchvision.datasets.mnist import MNIST
from torchnet.engine import Engine
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.init import kaiming_normal


def get_iterator(mode):
    ds = MNIST(root='./', download=True, train=mode)
    data = getattr(ds, 'train_data' if mode else 'test_data')
    labels = getattr(ds, 'train_labels' if mode else 'test_labels')
    tds = tnt.dataset.TensorDataset({'input': data, 'target': labels})
    return tds.parallel(batch_size=128, num_workers=4, shuffle=mode)


def conv_init(ni, no, k):
    return kaiming_normal(torch.Tensor(no, ni, k, k))


def linear_init(ni, no):
    return kaiming_normal(torch.Tensor(no, ni))


def f(params, inputs, mode):
    o = inputs.view(inputs.size(0), 1, 28, 28)
    o = F.conv2d(o, params['conv0.weight'], params['conv0.bias'], stride=2)
    o = F.relu(o)
    o = F.conv2d(o, params['conv1.weight'], params['conv1.bias'], stride=2)
    o = F.relu(o)
    o = o.view(o.size(0), -1)
    o = F.linear(o, params['linear2.weight'], params['linear2.bias'])
    o = F.relu(o)
    o = F.linear(o, params['linear3.weight'], params['linear3.bias'])
    return o


class ClassErrorHook(tnt.engine.Hook):

    def __init__(self, accuracy=False):
        super(ClassErrorHook, self).__init__()
        self.meter = tnt.meter.ClassErrorMeter(accuracy=accuracy)

    def on_start_epoch(self, state):
        self.meter.reset()

    def on_forward(self, state):
        self.meter.add(state['output'].data, torch.LongTensor(state['sample']['target']))


class LossMeterHook(tnt.engine.Hook):

    def __init__(self):
        super(LossMeterHook, self).__init__()
        self.meter = tnt.meter.AverageValueMeter()

    def on_start_epoch(self, state):
        self.meter.reset()

    def on_forward(self, state):
        self.meter.add(state['loss'])


class ProgbarHook(tnt.engine.Hook):

    def __init__(self, progbar):
        super(ProgbarHook, self).__init__()
        self.progbar = progbar

    def on_start_epoch(self, state):
        state['iterator'] = self.progbar(state['iterator'])


def main():
    params = {
            'conv0.weight': conv_init(1, 50, 5),  'conv0.bias': torch.zeros(50),
            'conv1.weight': conv_init(50, 50, 5), 'conv1.bias': torch.zeros(50),
            'linear2.weight': linear_init(800, 512), 'linear2.bias': torch.zeros(512),
            'linear3.weight': linear_init(512, 10),  'linear3.bias': torch.zeros(10),
            }
    params = {k: Variable(v, requires_grad=True) for k, v in params.items()}

    optimizer = torch.optim.SGD(params.values(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    engine = Engine(hook=[ClassErrorHook(accuracy=True), ProgbarHook(tqdm)])

    def h(state):
        sample = state['sample']
        mode = state['train']
        inputs = Variable(sample['input'].float() / 255.0)
        targets = Variable(sample['target'])
        o = f(params, inputs, mode)
        return F.cross_entropy(o, targets), o

    engine.train(closure=h, iterator=get_iterator(True), maxepoch=10, optimizer=optimizer)


if __name__ == '__main__':
    main()
