import torch
import wandb
from model import GRURNN
from dataset import MyDataset
from tokenizer import Tokenizer

wandb.init(project='gru_rnn')

# Initialize your dataset and dataloaders
# Initialize your model
model = GRURNN(input_size, hidden_size, output_size)
tokenizer = Tokenizer("path/to/spm.model")
dataset = MyDataset(data, tokenizer)
dataloader = torch.utils.data.DataLoader(dataset)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(n_epochs):
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # Training code here

        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss.item()})
