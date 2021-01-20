import neptune
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Set project
neptune.init(project_qualified_name='neptune-ai/tour-with-pytorch',
             api_token='ANONYMOUS')

PARAMS = {'lr': 0.005,
          'momentum': 0.9,
          'iterations': 100,
          'batch_size': 64}

# Create experiment
neptune.create_experiment(name='pytorch-run',
                          params=PARAMS)

train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data',
                                                          train=True,
                                                          download=True,
                                                          transform=transforms.Compose([transforms.ToTensor()])),
                                           batch_size=PARAMS['batch_size'],
                                           shuffle=True)

model = Net()
optimizer = optim.SGD(model.parameters(), PARAMS['lr'], PARAMS['momentum'])

# Training loop
for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    outputs = model(data)
    loss = F.nll_loss(outputs, target)

    # Log loss
    neptune.log_metric('batch_loss', loss)

    loss.backward()
    optimizer.step()

    # Log image predictions
    if batch_idx % 50 == 1:
        for image, prediction in zip(data, outputs):
            description = '\n'.join(['class {}: {}'.format(i, pred)
                                     for i, pred in enumerate(F.softmax(prediction))])
            neptune.log_image('predictions',
                              image.squeeze(),
                              description=description)

    if batch_idx == PARAMS['iterations']:
        break

# Log model weights
torch.save(model.state_dict(), 'model_dict.pth')
neptune.log_artifact('model_dict.pth')
