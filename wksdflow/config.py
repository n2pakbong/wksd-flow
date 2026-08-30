"""Every tunable number in one place, with automatic command-line overrides.

Each experiment calls Config.from_cli(**its_own_defaults):
  - dataclass defaults below  = package-wide fallback
  - kwargs passed by a script = that script's defaults
  - --flags on the command line = override both
So `python exp2_floor.py --n 64 --alpha-max 100` needs no file edits.
Adding a knob = adding one line to this dataclass.
"""
from dataclasses import dataclass, fields, asdict
import argparse, json, os


@dataclass
class Config:
    # ---- target: pi ∝ exp(-V / eps^2) ------------------------------------
    dim: int = 2
    eps: float = 1.0        # the epsilon of the paper
    scale: float = 1.0      # gaussian_target: V = |x|^2 / (2 scale^2)
    sep: float = 0.9        # mixture_target: modes at ±sep·e_1
    n_data: int = 200       # logistic_posterior: number of observations

    # ---- kernel (Prop. admissible_class) ---------------------------------
    base: str = "imq"       # imq | gaussian | rq | matern52 | matern72 | matern92
    ell: float = 1.0        # length scale
    s: float = 0.0          # weight w(x) = (1+|x|^2)^{-s/2}
    delta: float = 1e-6     # sqrt shift, radial (matern) kernels only
    ell_q: float = 1.0      # length scale of the collocation kernel q

    # ---- resolvent (Def. resolvent_preconditioner) -----------------------
    alpha: float = 1.0      # (-L + alpha) Psi = h_mu
    gamma: float = 1e-8     # Tikhonov ridge, Algorithm 2

    # ---- particles / time stepping ---------------------------------------
    n: int = 200
    eta: float = 5e-2
    n_steps: int = 2000
    seed: int = 0

    # ---- sweeps ----------------------------------------------------------
    alpha_min: float = 1.5      # exp2, log-spaced
    alpha_max: float = 30.0
    n_alpha: int = 7
    plateau_window: int = 500    # exp2: how many final steps to average
    n_list: str = "64,128,256,512"   # exp3

    # ---- output ----------------------------------------------------------
    outdir: str = "figures"
    fmt: str = "png"        # png previews inline in Codespaces; pdf for the paper
    tag: str = ""           # suffix on file names, e.g. --tag _run7
    diag: int = 1           # 1 = print diagnostics before integrating

    # ---- helpers ---------------------------------------------------------
    @property
    def n_values(self):
        return [int(v) for v in self.n_list.split(",")]

    def kernel_kwargs(self):
        kw = {"ell": self.ell}
        if self.base.startswith("matern"):
            kw["delta"] = self.delta
        return kw

    def path(self, stem):
        os.makedirs(self.outdir, exist_ok=True)
        return f"{self.outdir}/{stem}{self.tag}.{self.fmt}"

    @classmethod
    def from_cli(cls, argv=None, **script_defaults):
        base = cls(**script_defaults)
        p = argparse.ArgumentParser()
        for f in fields(cls):
            d = getattr(base, f.name)
            p.add_argument("--" + f.name.replace("_", "-"), dest=f.name,
                           type=type(d), default=d)
        cfg = cls(**vars(p.parse_args(argv)))
        print("config:", json.dumps(asdict(cfg), indent=2), flush=True)
        return cfg
