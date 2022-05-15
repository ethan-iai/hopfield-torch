import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.nn import Parameter
from collections import defaultdict

def train(model, 
          train_loader, 
          selected_label_indices, 
          train_data_size=None):
    selected_label_indices = set(selected_label_indices)
    sample_count = 0
    for data, target in tqdm(train_loader, desc='Train'):
        data.squeeze_()
        if target.item() not in selected_label_indices:
            continue
        # FIXME: great risk to overflow, try to adopt momentun update instead 
        
        model.weight += torch.outer(data, data)
        if model.bias is not None:
            # TODO: 
            raise NotImplemented

        sample_count += 1
        
        if train_data_size is not None and \
           sample_count >= train_data_size:
           break

    model.weight /= sample_count     
    model.weight -= torch.diag(model.weight)

    if model.bias is not None:
        model.bias /= sample_count

def test(model, 
         test_loader, 
         selected_label_indices, 
         sample_per_class=None):
    if sample_per_class is None:
        sample_per_class = 2 * len(selected_label_indices)

    inputs = defaultdict(list)
    outputs = defaultdict(list)
    
    counts = {label_idx: 0 for label_idx in selected_label_indices}
    selected_label_indices = set(selected_label_indices)
    for data, target in test_loader:
        data.squeeze_()
        target = target.item()
        if all([cnt >= sample_per_class for cnt in counts.values()]):
            break
        if target not in counts or \
           counts.get(target, -1) >= sample_per_class:
            continue

        output = model(data)
        inputs[target].append(data)
        outputs[target].append(output)

        counts[target] += 1

    return inputs, outputs


class HopfieldNet(nn.Module):

    def __init__(self, 
                 in_features,
                 bias=True,
                 threshold=0.0, 
                 max_iter=128):
        super(HopfieldNet, self).__init__()

        self.in_features = in_features
        self.threshold = threshold
        self.max_iter = max_iter

        self.weight = Parameter(torch.zeros(in_features, in_features), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)

    def _energy(self, x):
        e = -0.5 * x @ self.weight @ x
        if self.bias is not None:
            e -= self.bias @ x
        return e
        
    def _run(self, x, eps=1e-6):
        """Synchronousl update

        Args:
            x (torch.Tensor): inputs
            eps (float): Defaults to 1e-6.

        """
        from torchvision.utils import save_image
        e = self._energy(x)

        for _ in range(self.max_iter):

            x = torch.sign(
                F.linear(x, self.weight, self.bias) 
                    - self.threshold)

            new_e = self._energy(x)
            if abs(new_e - e) < eps:
                return x

            e = new_e
        return x

    def forward(self, x):
        assert x.ndim == 1
        
        return self._run(x)
    
    def extra_repr(self):
        return 'in_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

