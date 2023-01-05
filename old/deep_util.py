import datetime
import os

import numpy as np
import torch

import old.puzzle as puzzle


# TODO: Add transformation code
class KenKenDataset(torch.utils.data.Dataset):
    """KenKenDataset of n by n puzzles by size size"""

    def __init__(self, root_dir: str=None, n: int=5, size: int=1000):
        if root_dir:
            self.root_dir = root_dir
            self.X = torch.load(os.path.join(self.root_dir, 'X.pt'))
            self.Y = torch.load(os.path.join(self.root_dir, 'Y.pt'))
        else:
            X_path, y_path = self._generate_new_data(n=n, size=size)
            self.X = torch.load(X_path)
            self.Y = torch.load(y_path)
    
    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]

    def _generate_new_data(self, n: int, size: int):
        self.root_dir = 'data/%s-%s-%s' % (
            n,
            size,
            datetime.datetime.now().strftime('%S-%M-%H-%d-%m-%Y')
        )
        os.makedirs(self.root_dir)

        print('Generating new dataset of n=%s and size=%s' % (n, size))
        print('New dataset root directory: %s' % self.root_dir)

        X, Y = [], []
        for _ in range(size):
            if _ % 1000 == 0: print('Round: %s' % _)

            p = puzzle.KenKen(n)
            x = torch.dstack((
                torch.nn.functional.one_hot(
                    torch.tensor(p._cages), # this won't work because of a variable input size...fuck
                    num_classes=len(p._dict_rep)
                ),
                torch.nn.functional.one_hot(
                    torch.tensor(p._operators),
                    num_classes=5
                ),
                torch.tensor(p._totals) # these should be normalized against possible max
            )).float()
            y = torch.nn.functional.one_hot(
                torch.tensor(p._answers_machine),
                num_classes=n,
            ).float()
            X.append(x)
            Y.append(y)

        X_path = os.path.join(self.root_dir, 'X.pt')
        Y_path = os.path.join(self.root_dir, 'Y.pt')
        torch.save(X, X_path)
        torch.save(Y, Y_path)

        return X_path, Y_path


class ANNModel(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.l0 = torch.nn.Linear(3, 50)
        self.l1 = torch.nn.Linear(50, 50)
        self.l2 = torch.nn.Linear(50, 50)
        self.l3 = torch.nn.Linear(50, self.n)

    def forward(self, x):
        x = torch.nn.functional.relu(self.l0(x))
        x = torch.nn.functional.relu(self.l1(x))
        x = torch.nn.functional.relu(self.l2(x))
        x = torch.nn.functional.relu(self.l3(x))
        return x


class Trainer():
    def __init__(self, train_dataloader, test_dataloader, model, loss_fn, optimizer):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_loop(self):
        size = len(self.train_dataloader.dataset)
        for batch, (X, y) in enumerate(self.train_dataloader):
            print('Training batch: ' % batch)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(self):
        size = len(self.test_dataloader.dataset)
        print(size)
        num_batches = len(self.test_dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in self.test_dataloader:
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                print('next try')
                print(y.argmax(3))
                print(pred)
                print(pred.argmax(3))
                correct += 1 if (pred.argmax(3) == y.argmax(3)).all() else 0
        
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")