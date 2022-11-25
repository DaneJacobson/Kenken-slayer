import torch

import deep_util

n = 4
train_dataloader = torch.utils.data.DataLoader(
    deep_util.KenKenDataset(n=n, size=10000),
    batch_size=100
)
test_dataloader = torch.utils.data.DataLoader(
    deep_util.KenKenDataset(n=n, size=1000),
    batch_size=1
)
model = deep_util.ANNModel(n=n)
loss_fn = torch.nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

trainer = deep_util.Trainer(
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    model=model,
    loss_fn=loss_fn,
    optimizer=optim
)

trainer.train_loop()
trainer.test_loop()

torch.save(model.state_dict(), 'model/bigboi.pt')