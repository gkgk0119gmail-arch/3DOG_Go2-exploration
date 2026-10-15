"""Selector hook of the RL head: samples (training) or picks greedily (evaluation) and records every decision."""

from __future__ import annotations

import torch

from .features import observe, to_batch


class Collector:
    """Selector hook for every robot of one session: samples actions and records the decisions."""

    def __init__(self, net, device, greedy=False):
        self.net, self.dev, self.greedy = net, device, greedy
        self.session = None
        self.dec = {}   # env -> list of dicts

    def factory(self, e):
        self.dec[e] = []

        def select(agent, targets, times, t):
            obs = observe(agent, targets, times, t)
            cand, mask, glob = to_batch([obs], self.dev)
            with torch.no_grad():
                logits, v = self.net(cand, mask, glob)
            dist = torch.distributions.Categorical(logits=logits[0])
            a = int(logits[0].argmax()) if self.greedy else int(dist.sample())
            co = float(self.session.surf.co()[e])
            self.dec[e].append({"obs": obs, "a": a, "logp": float(dist.log_prob(torch.tensor(a, device=self.dev))),
                                "v": float(v[0]), "t": t, "co": co, "n": len(targets)})
            return a if a < len(targets) else -1

        return select


def load_policy(path: str, cfg, device: str = "cuda:0"):
    """A trained InspectionPolicy from a checkpoint of scripts/pose_inspection/rl_train.py (eval mode)."""
    from .policy import InspectionPolicy

    ck = torch.load(path, map_location=device)
    rl = ck.get("cfg") or cfg.rl
    get = (lambda k: rl[k]) if isinstance(rl, dict) else (lambda k: rl.get(k))
    net = InspectionPolicy(get("d_model"), get("heads"), get("layers")).to(device)
    net.load_state_dict(ck["net"])
    net.eval()
    return net
