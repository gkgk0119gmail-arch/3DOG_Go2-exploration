"""Networks with the extra node inputs, warm-started from the original ARiADNE checkpoint."""

import torch

from model import PolicyNet, QNet
from parameter import EMBEDDING_DIM, NODE_INPUT_DIM


def _widen(state, key, new_in):
    """Pad the first layer's input columns with zeros so the new inputs start out ignored."""
    w = state[key]
    if w.shape[1] < new_in:
        state[key] = torch.cat([w, torch.zeros(w.shape[0], new_in - w.shape[1], dtype=w.dtype)], dim=1)
    return state


def load_pretrained(path, device="cpu"):
    """Returns policy, q1, q2 (and log_alpha) from an original (4 / 5 input) or our (6 / 7 input) checkpoint."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    policy = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
    q1, q2 = QNet(NODE_INPUT_DIM + 1, EMBEDDING_DIM), QNet(NODE_INPUT_DIM + 1, EMBEDDING_DIM)
    policy.load_state_dict(_widen(dict(ck["policy_model"]), "initial_embedding.weight", NODE_INPUT_DIM))
    if "q_net1_model" in ck:  # policy-only snapshots (policy_ep*.pth) have no critics
        q1.load_state_dict(_widen(dict(ck["q_net1_model"]), "initial_embedding.weight", NODE_INPUT_DIM + 1))
        q2.load_state_dict(_widen(dict(ck["q_net2_model"]), "initial_embedding.weight", NODE_INPUT_DIM + 1))
    log_alpha = ck.get("log_alpha", torch.tensor([-2.0]))
    return policy.to(device), q1.to(device), q2.to(device), log_alpha.detach().clone().to(device), ck
