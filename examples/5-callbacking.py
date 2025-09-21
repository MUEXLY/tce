from pathlib import Path
import os

import numpy as np
import requests
from ase import build

from tce.training import ClusterExpansion
from tce.monte_carlo import monte_carlo


def discord_webhook_callback(
        step: int,
        num_steps: int,
        env_var: str = "DISCORD_WEBHOOK_URL",
        message: str = "CuNi mc run finished"
):

    if not os.getenv(env_var) or step + 1 < num_steps:
        return

    response = requests.post(url=os.getenv(env_var), json={"content": message})
    response.raise_for_status()


def main():

    rng = np.random.default_rng(seed=0)

    cluster_expansion = ClusterExpansion.load(Path("CuNi.pkl"))

    atoms = build.bulk(
        cluster_expansion.type_map[0],
        a=cluster_expansion.cluster_basis.lattice_parameter,
        crystalstructure=cluster_expansion.cluster_basis.lattice_structure.name.lower(),
        cubic=True
    ).repeat((10, 10, 10))
    atoms.symbols = rng.choice(cluster_expansion.type_map, size=len(atoms))

    trajectory = monte_carlo(
        initial_configuration=atoms,
        cluster_expansion=cluster_expansion,
        num_steps=10_000,
        beta=19.341,
        save_every=100,
        callback=discord_webhook_callback,
    )

if __name__ == "__main__":

    main()
