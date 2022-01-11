import torch
import torch.nn as nn
from torch.distributions.utils import logits_to_probs

from src.common.utils import init


class FixedCategorical:

    def __init__(self, logits):
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        probs = logits_to_probs(logits)

        self.logits = logits

        self.probs = probs

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

    def sample(self):
        return torch.multinomial(self.probs, 1, True).T

    def log_probs(self, action):
        val, log_pmf = torch.broadcast_tensors(action, self.logits)
        val = val[..., :1]
        action_log_probs = log_pmf.gather(-1, val).squeeze(-1)
        action_log_probs = action_log_probs.view(action.size(0), -1).sum(-1).unsqueeze(-1)

        return action_log_probs

