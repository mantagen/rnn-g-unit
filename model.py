import torch
import torch.nn as nn

class GRURNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRURNN, self).__init__()

        self.hidden_size = hidden_size

        # GRU cell components
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size)

        # Output layer
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), 1)

        reset = torch.sigmoid(self.reset_gate(combined))
        update = torch.sigmoid(self.update_gate(combined))

        combined_reset = torch.cat((x, reset * hidden), 1)
        candidate = torch.tanh(self.candidate(combined_reset))

        hidden = (1 - update) * candidate + update * hidden
        output = self.out(hidden)

        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)
