import torch
from torch import nn


class ActorNet(nn.Module):
    """PyTorch replacement for the legacy TF1 stacked-LSTM actor.

    Input state shape: [batch, time_steps, 3] with columns
    [price, utilization_or_sales, remaining_time].
    """

    def __init__(
        self,
        state_dim: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        c_dim: int = 128,
        action_range: float = 40.0,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.action_range = float(action_range)
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size + c_dim, 1)
        self._init_forget_bias(0.6)

    def _init_forget_bias(self, value: float):
        # TF BasicLSTMCell used forget_bias=0.6. PyTorch LSTM has two
        # bias tensors per layer; split the desired total bias across both.
        for name, param in self.lstm.named_parameters():
            if "bias_ih" in name or "bias_hh" in name:
                n = param.numel()
                start, end = n // 4, n // 2
                with torch.no_grad():
                    param[start:end].fill_(value / 2.0)

    def forward(self, state, c=None):
        out, _ = self.lstm(state)
        last = out[:, -1, :]
        if c is None:
            c = torch.zeros(
                (state.shape[0], self.c_dim),
                dtype=state.dtype,
                device=state.device,
            )
        x = torch.cat([last, c], dim=1)
        return torch.tanh(self.head(x)) * self.action_range
