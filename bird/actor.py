import torch

from .rl.model import ActorNet


class TorchActor:
    def __init__(self, checkpoint, device='cpu'):
        ckpt = torch.load(checkpoint, map_location=device)
        self.device = torch.device(device)
        self.model = ActorNet(
            state_dim=ckpt.get('state_dim', 3),
            hidden_size=ckpt.get('hidden_size', 128),
            num_layers=ckpt.get('num_layers', 2),
            c_dim=ckpt.get('c_dim', 128),
            action_range=ckpt.get('action_range', 40.0),
        ).to(self.device)
        state = ckpt.get('actor_state', ckpt.get('model_state'))
        if state is None:
            raise KeyError('checkpoint has neither actor_state nor model_state')
        self.model.load_state_dict(state)
        self.model.eval()

    def action(self, state, c):
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device)[None, ...]
        cc = torch.as_tensor(c, dtype=torch.float32, device=self.device)[None, ...]
        with torch.no_grad():
            return float(self.model(x, cc).cpu().item())
