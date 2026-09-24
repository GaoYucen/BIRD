import torch
from torch import nn


class ActorNet(nn.Module):
    """Stacked-LSTM actor for BIRD's RL pricing strategy."""

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


class CriticNet(nn.Module):
    """PyTorch counterpart of the legacy stacked-LSTM critic.

    The old TF critic embeds the scalar action into a c_dim vector, concatenates
    it with the final LSTM output and the c vector, applies a 64-unit hidden
    layer, and predicts one Q value.
    """

    def __init__(
        self,
        state_dim: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        c_dim: int = 128,
        action_dim: int = 1,
        hidden_fc: int = 64,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.action_embed = nn.Linear(action_dim, c_dim)
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_size + c_dim + c_dim, hidden_fc)
        self.fc2 = nn.Linear(hidden_fc, 1)
        self._init_forget_bias(0.6)

    def _init_forget_bias(self, value: float):
        for name, param in self.lstm.named_parameters():
            if "bias_ih" in name or "bias_hh" in name:
                n = param.numel()
                start, end = n // 4, n // 2
                with torch.no_grad():
                    param[start:end].fill_(value / 2.0)

    def forward(self, state, c, action):
        out, _ = self.lstm(state)
        last = out[:, -1, :]
        action_embed = self.action_embed(action)
        x = torch.cat([last, action_embed, c], dim=1)
        x = self.fc1(x)
        return self.fc2(x)
