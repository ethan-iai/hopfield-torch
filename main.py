import os
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from hopfield import HopfieldNet, train, test
from utils import Binarize, Flatten, linear_unscale

SAVE_DIR = '.'
SELECTED_LABEL_INDICES = [0, 4, 8]
SAMPLE_PER_CLASS = 2 * len(SELECTED_LABEL_INDICES)
TRAIN_DATA_SIZE = None

def main():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
            transform=transforms.Compose([
                Binarize(),
                transforms.ToTensor(),
                Flatten()
            ])), 
        batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True,
            transform=transforms.Compose([
                Binarize(),
                transforms.ToTensor(),
                Flatten()
            ])), 
        batch_size=1, shuffle=False)

    model = HopfieldNet(784, bias=False)

    with torch.no_grad():
        train(model, train_loader, 
              SELECTED_LABEL_INDICES, TRAIN_DATA_SIZE)
        inputs, outputs = test(model, test_loader, 
                               SELECTED_LABEL_INDICES,
                               SAMPLE_PER_CLASS)
    
    imgs = []
    for label in SELECTED_LABEL_INDICES: 
        imgs.extend(inputs[label])
        imgs.extend(outputs[label])
        # for i, output in enumerate(outputs[label]):
        #     save_image(linear_unscale(output).reshape(1, 28, 28), f'{label}_{i}.png')
    
    imgs = list(map(lambda tensor: linear_unscale(tensor).reshape((1, 28, 28)), imgs))

    # torch.save(
    #     { 'params': model.state_dict() }, 
    #     os.path.join(SAVE_DIR, f'checkpoint_{TRAIN_DATA_SIZE}.pth')
    # )
    save_image(
        make_grid(imgs, nrow=SAMPLE_PER_CLASS, pad_value=1.0),
        os.path.join(SAVE_DIR, f'test_result_{TRAIN_DATA_SIZE}.png')
    )
    save_image(
        linear_unscale(model.weight.data),
        os.path.join(SAVE_DIR, f'model_weights_{TRAIN_DATA_SIZE}.png') 
    )
    
if __name__ == '__main__':
    # for size in [5, 10, 100, 1000, 10000, None]:
    #     TRAIN_DATA_SIZE = size
    #     main()
    main()