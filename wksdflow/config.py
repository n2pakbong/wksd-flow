"""One dataclass holding every tunable number, plus automatic CLI parsing.

Each experiment script calls Config.from_cli(**its_own_defaults). The dataclass
defaults below are the package-wide fallbacks; the kwargs a script passes are
that script's defaults; and anything on the command line overrides both. So
    python exp1_rates.py --n 64 --n-steps 300 --alpha 0.5 --base gaussian
works with no edits to the script. Print cfg at the top of every run and paste
it into the figure caption; that is your reproducibility record.
"""
from dataclasses import dataclass, fields, asdict
import argparse, json


@dataclass
class Config:
    # ---- target: pi ∝ exp(-V/eps^2) --------------------------------------
    dim: int = 2
    eps: float = 1.0          # the epsilon of the paper, temperature scale
    scale: float = 1.0        # gaussian_target: V = |x|^2/(2 scale^2)
    sep: float = 0.9          # mixture_target: modes at ±sep e_1
    n_data: int = 200         # logistic_posterior: number of observations

    # ---- base kernel and the weighting of Prop. (admissible_class) -------
    base: str = "imq"         # imq | gaussian | rq | matern52/72/92
    ell: float = 1.0          # length scale
    s: float = 0.0            # weight w(x) = (1+|x|^2)^{-s/2}
    delta: float = 1e-6       # sqrt shift, radial kernels only

    # ---- resolvent preconditioner (Def. resolvent_preconditioner) --------
    alpha: float = 1.0        # (-L + alpha) Psi = h_mu
    gamma: float = 1e-8       # Tikhonov ridge in Algorithm 2

    # ---- particles and time stepping ------------------------------------
    n: int = 200
    eta: float = 5e-2
    n_steps: int = 2000
    seed: int = 0

    # ---- sweeps ----------------------------------------------------------
    alpha_min: float = 1.0    # exp2: alpha sweep endpoints (log spaced)
    alpha_max: float = 30.0
    n_alpha: int = 7
    n_list: str = "64,128,256,512"   # exp3: particle counts

    # ---- output ----------------------------------------------------------
    outdir: str = "figures"
    fmt: str = "png"          # png previews in Codespaces; use pdf for the paper
    tag: str = ""             # suffix appended to file names

    @property
    def n_values(self):
        return [int(v) for v in self.n_list.split(",")]

    def kernel_kwargs(self):
        """base_kwargs for make_kernel_bundle: delta only for radial kernels."""
        kw = {"ell": self.ell}
        if self.base.startswith("matern"):
            kw["delta"] = self.delta
        return kw

    @classmethod
    def from_cli(cls, argv=None, **script_defaults):
        base = cls(**script_defaults)
        p = argparse.ArgumentParser(
            description="Override any field of Config from the command line.")
        for f in fields(cls):
            p.add_argument("--" + f.name.replace("_", "-"), dest=f.name,
                           type=f.type, default=getattr(base, f.name))
        cfg = cls(**vars(p.parse_args(argv)))
        print("config:", json.dumps(asdict(cfg), indent=2))
        return cfg

    def path(self, stem):
        import os
        os.makedirs(self.outdir, exist_ok=True)
        return f"{self.outdir}/{stem}{self.tag}.{self.fmt}"
