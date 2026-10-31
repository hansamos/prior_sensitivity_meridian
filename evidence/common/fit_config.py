"""Settings shared by the fit and refit scripts: reference prior, MCMC modes, and run metadata helpers."""
import os
import platform
import sys

import arviz as az
import jax
import numpy as np
import pandas as pd

import meridian

BASE_MU, BASE_SIGMA = 0.2, 0.9


class HashableDict(dict):
    """Meridian 2.1.0 (JAX backend) jit-compiles NUTS with `dual_averaging_kwargs` as a static
    argument, so a plain dict raises 'Non-hashable static arguments are not supported'."""

    def __hash__(self):
        return hash(tuple(sorted(self.items())))


# "strong" is the 40-geo x 156-week model used in all regional experiments; "cpu_pilot" is a small, fast variant.
MODES = {
    "cpu_pilot": dict(n_geos=3, n_times=80, enable_aks=False,
                      mcmc=dict(n_chains=2, n_adapt=300, n_burnin=100, n_keep=300)),
    "strong": dict(n_geos=None, n_times=None, enable_aks=True,
                   mcmc=dict(n_chains=4, n_adapt=1000, n_burnin=500, n_keep=1000)),
}
SMOKE_MCMC = dict(n_chains=2, n_adapt=60, n_burnin=20, n_keep=60)


def hardware():
    cpu = ""
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:  # noqa: BLE001
        pass
    return dict(
        node=platform.node(), platform=platform.platform(), cpu_model=cpu,
        os_cpu_count=os.cpu_count(),
        slurm={k: v for k, v in os.environ.items() if k.startswith("SLURM_") and k in (
            "SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE",
            "SLURM_JOB_NODELIST")},
        jax_devices=[str(d) for d in jax.devices()],
    )


def versions():
    import scipy, xarray, tensorflow as tf  # noqa: E401
    import tensorflow_probability as tfp
    return dict(python=sys.version.split()[0], meridian=meridian.__version__, arviz=az.__version__,
                jax=jax.__version__, tensorflow=tf.__version__, tfp=tfp.__version__,
                numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
                xarray=xarray.__version__, jax_enable_x64=bool(jax.config.jax_enable_x64))
