r"""
.. include:: ../README.md

# Examples

## ⚛️ Using Atomic Simulation Environment (ASE)

Below is an example of converting an `ase.Atoms` object into a feature vector $\mathbf{t}$. The mapping is not exactly
one-to-one, since an `ase.Atoms` object sits on a dynamic lattice rather than a static one, but we can regardless
provide `tce-lib` sufficient information to compute $\mathbf{t}$. The code snippet below uses the version `ase==3.26.0`.

```py
.. include:: ../examples/using-ase.py
```

## 💎 Exotic Lattice Structures

Below is an example of injecting a custom lattice structure into `tce-lib`. To do this, we must extend the
`LatticeStructure` class, which we will do using [aenum](https://pypi.org/p/aenum/) (version `aenum==3.1.16`
specifically). We use a cubic diamond structure here as an example, but this extends to any atomic basis in any
tetragonal unit cell.

```py
.. include:: ../examples/exotic-lattice.py
```

We are also more than happy to include new lattice types as native options in `tce-lib`! Please either open an issue
[here](https://github.com/MUEXLY/tce-lib/issues), or a pull request [here](https://github.com/MUEXLY/tce-lib/pulls) if
you are familiar with GitHub.

If your unit cell is available via `ase`'s functionalities (below is using `ase.build.bulk`, but you can also load
in a `.cif` file or something similar), you don't have to waste your time finding atomic bases and cutoffs every time.
For example, for a [fluorite structure](https://en.wikipedia.org/wiki/Fluorite_structure) with U and Th cations:

```py
.. include:: ../examples/exotic-lattice2.py
```

You'll notice that a lot of the features are $0$. This is not uncommon for exotic lattice types, especially when not
all lattice sites are equivalent. This is not a problem - we just likely need to feature reduce later using something
like [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis), which is relatively easy with an
`sklearn.pipeline.Pipeline` object (docs are
[here](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)). This will be elaborated
upon in an example more focused on training ([here](https://muexly.github.io/tce-lib/tce.html#custom-training-advanced)).

## 🔩 FeCr + EAM (basic)

Below is a very basic example of computing a best-fit interaction vector from LAMMPS data. We use LAMMPS and an EAM
potential from Eich et al. (paper [here](https://doi.org/10.1016/j.commatsci.2015.03.047)), use `tce-lib` to build a
best-fit interaction vector from a sequence of random samples, and cross-validate the results using `scikit-learn`.

```py
.. include:: ../examples/iron-chrome-lammps.py
```

Above, we avoided a lot of convenience functions, like constructing a feature vector calculator from a feature vector
computer factory in the prior examples. You can very easily still use these factories, but the above example shows that
you should not feel obligated to use them for more advanced use cases.

This generates the plot below:

[<img
    src="https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/cross-val.png"
    width=100%
    alt="Residual errors during cross-validation"
    title="Residual errors"
/>](https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/cross-val.png)

The errors are not great here (a good absolute error is on the order of 1-10 meV/atom as a rule of thumb). The fit
would be much better if we included partially ordered samples as well. We emphasize that this is a very basic example,
and that a real production fit should be done against a more diverse training set than just purely random samples.

This example serves as a good template for using programs other than LAMMPS to compute energies. For example, one could
define a constructor that creates a `Calculator` instance that wraps VASP:

```py
from ase.calculators.vasp import Vasp

calculator_constructor = lambda: Vasp(
    prec="Accurate",
    encut=500,
    istart=0,
    ismear=1,
    sigma=0.1,
    nsw=400,
    nelmin=5,
    nelm=100,
    ibrion=1,
    potim=0.5,
    isif=3,
    isym=2,
    ediff=1e-5,
    ediffg=-5e-4,
    lreal=False,
    lwave=False,
    lcharg=False
)
```

See ASE's documentation [here](https://ase-lib.org/ase/calculators/vasp.html) for how to properly set this up!

## 🏋️‍♀️ Training + Monte Carlo

Below is a slightly more involved example of creating a model and deploying it for a Monte Carlo run.

This showcases two utility modules, namely `tce.training` and `tce.monte_carlo`. These mostly contain wrappers, so
feel free to avoid them! If you are using this for a novel research idea, it is likely that these wrappers are too
basic (which is a good thing for you!).

The first script is training a CuNi model using an EAM potential from Fischer et al.
(paper [here](https://doi.org/10.1016/j.actamat.2019.06.027)). In this script, we generate a bunch of random CuNi
solid solutions, attach an `ase.calculators.eam.EAM` calculator to each configuration, compute their energies, and
then train using the `tce.training.train` method, which returns a `tce.training.ClusterExpansion` instance. The
container is then saved to be used for later.

**IMPORTANT**: These are unrelaxed energies! A real production environment should optimize the structure - see the
prior example on how to do this within a LAMMPS calculator.

```py
.. include:: ../examples/0-copper-nickel-training.py
```

The next script uses the saved container to run a canonical Monte Carlo simulation on a $10\times 10\times 10$
supercell, storing the configuration (saved in an `ase.Atoms` object) every 100 frames. We also set up a `logging`
configuration here, which will tell you how far-along the simulation is. Note that `trajectory` looks complicated, but
is just a list of `ase.Atoms` objects, so you have a lot of freedom to do what you wish with this trajectory later.

```py
.. include:: ../examples/1-copper-nickel-mc.py
```

These are then visualizable with a number of softwares, including [OVITO](https://www.ovito.org/). An example of such a
rendering is below:

<div style="padding:50% 0 0 0;position:relative;"><iframe src="https://player.vimeo.com/video/1117980384?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479" frameborder="0" allow="autoplay; fullscreen; picture-in-picture; clipboard-write; encrypted-media; web-share" referrerpolicy="strict-origin-when-cross-origin" style="position:absolute;top:0;left:0;width:100%;height:100%;" title="animation"></iframe></div><script src="https://player.vimeo.com/api/player.js"></script>

Just from the animation, it doesn't look like much is happening at all. The animation is not the whole story, though -
you can also use the trajectory to do some analysis. We can use OVITO's Python library [here](https://pypi.org/p/ovito)
and any of its plugins to do some analysis, as if our files are from any other atomistic simulation software. Below
we'll compute the Cowley short range order parameter using the
`cowley-sro-parameters` plugin [here](https://pypi.org/p/cowley-sro-parameters) (shameless plug... I'm the author 🙂).

```py
.. include:: ../examples/2-copper-nickel-sro.py
```

This generates the plot below. A negative value indicates attraction between two atom types. So, the solution is
clearly not fully random! We probably need a lot more than 10,000 steps too - this curve should bottom out once we
reach steady state. Note we can also just grab the potential energy from the `ase.Atoms` instances - the Monte Carlo
run stores this information using `ase.calculators.singlepoint.SinglePointCalculator` instances.

[<img
    src="https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/cu-ni-sro.png"
    width=100%
    alt="CuNi SRO parameter from CE"
    title="SRO parameter"
/>](https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/cu-ni-sro.png)

We can also use the model to sample a different ensemble. For the prototypical canonical ensemble, the acceptance rule
for a swap with energy difference $\Delta E$ is $\exp(-\beta\Delta E) > u$, where $u$ is a random number drawn from
$[0, 1]$. For the grand canonical ensemble, the acceptance rule is instead:

$$ \exp\left(-\beta\left(\Delta E - \sum_\alpha \mu_\alpha \Delta N_\alpha\right)\right) = \exp\left(-\beta\left(\Delta E - \boldsymbol{\mu}\cdot\Delta \mathbf{N}\right)\right) > u $$

where $\mu_\alpha$ is the chemical potential of type $\alpha$ and $\Delta N_\alpha$ is the change in the number of
$\alpha$ atoms in the swap. You can inject this into `tce.monte_carlo.monte_carlo` by defining an `energy_modifier`,
which adds a term to $\Delta E$:

```py
.. include:: ../examples/1-copper-nickel-mc2.py
```

We also can specify our own Monte Carlo step, which is done above. This plot generates a curve which is useful for
computing phase diagrams:

[<img
    src="https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/cu-ni-sgcmc.png"
    width=100%
    alt="CuNi SGCMC curve"
    title="CuNi curve"
/>](https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/cu-ni-sgcmc.png)

Note that the curve is continuous, which denotes no phase transitions at the temperature. This matches experimental
phase diagrams - CuNi forms a solid solution along the whole composition range below the melting point.

<div style="text-align: center;">
  <a href="https://sv.rkriz.net/classes/MSE2094_NoteBook/96ClassProj/OLD/examples/cu_ni.jpg">
    <img
      src="https://sv.rkriz.net/classes/MSE2094_NoteBook/96ClassProj/OLD/examples/cu_ni.jpg"
      width="50%"
      alt="CuNi phase diagram"
      title="CuNi"
    />
  </a>
</div>

Image credit: Ron Kriz, MSE 2094 @ Virginia Tech ([url](https://sv.rkriz.net/classes/MSE2094_NoteBook/96ClassProj/OLD/examples/cu-ni.html))

## 💻 Custom Training (Advanced)

Below is an example of using a custom training method to train the CE model. There are many reasons one might want to do
this. The example below is a very typical one - using [lasso](https://en.wikipedia.org/wiki/Lasso_(statistics)). This
regularization technique minimizes the loss:

$$ L(\beta\; |\;\lambda) = \|X\beta - y\|_2^2 + \lambda \|\beta\|_1 $$

which better-supports sparse best-fit parameters $\hat{\beta}$, which may be useful if you only want to exclude
non-important clusters. We'll use `scikit-learn`'s interface for providing a model. You can really use any linear
model here (without an intercept...), see `scikit-learn`'s docs
[here](https://scikit-learn.org/stable/modules/linear_model.html) for more examples of these.

```py
.. include:: ../examples/3-sklearn-fitting.py
```

This script (it will be quite slow...) will calculate the number of nonzero cluster interaction coefficients as a
function of the regularization parameter. For larger regularization parameters, the number of nonzero coefficients
should decrease.

[<img
    src="https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/regularization.png"
    width=100%
    alt="Lasso regularization"
    title="Lasso"
/>](https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/regularization.png)


## 🧲 Learning a tensorial property

In general, one might also want to learn tensorial properties. This can be done by vectorizing the property in some
way, like [Voigt notation](https://en.wikipedia.org/wiki/Voigt_notation):

$$ \sigma = (\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{yz}, \sigma_{xz}, \sigma_{xy}) $$

Below is an example of changing the target property to stress rather than energy. It also showcases an important point
about `tce-lib`: our feature vectors are **extensive**, not intensive like other CE libraries. This matters when
training on intensive properties, like stress. Here, we need to actually train on an "extensive" version of stress, and
we can just make it intensive later. Of course, it is also fine to use this same pattern to train a CE model on
other scalar properties.

```py
.. include:: ../examples/4-tensorial-property.py
```

## 🔔 Callback functionality

The `tce.monte_carlo.monte_carlo` routine also has a `callback` argument that lets you inject a notification system
into the Monte Carlo run. This argument needs to be a function with signature:

```py
def callback(step: int, num_steps: int) -> None:
    ...
```

If it is not provided, it defaults to calling the `logging` library:

```py
import logging

LOGGER = logging.getLogger(__name__)

def callback(step_: int, num_steps_: int):
    LOGGER.info(f"MC step {step_:.0f}/{num_steps_:.0f}")
```

But, you can do very cool things with this. It's a bit of a cute example that might not be practical, but you can
send notifications to third party systems like [Discord](https://en.wikipedia.org/wiki/Discord) using webhooks.

```py
.. include:: ../examples/5-callbacking.py
```

which will send a notification in whatever Discord channel once the MC run is finished. See
[Discord's documentation on webhooks](https://support.discord.com/hc/en-us/articles/228383668-Intro-to-Webhooks) for
a tutorial on how to set up your own webhook URL. You can get really creative here too, like Slack's similar
functionality [here](https://docs.slack.dev/messaging/sending-messages-using-incoming-webhooks/), or the
[Gmail API](https://developers.google.com/workspace/gmail/api/guides). None of these are particularly useful for what I
have done above (sending a single email once the run is finished), but really shine for long runs where you want to
be periodicially notified.

## 🕵️ Loading and Visualizing Datasets

Below is an example of using one of our pre-set training datasets using `tce.datasets`. When you install `tce-lib`, you
automatically install some toy datasets that are mostly of pedagogical benefit, i.e. you can look at one of these datasets 
and see examples of what you can train on. Since `ovito` has an `ase` interface, you can also use `ovito` to visualize the 
dataset, which might be of interest.

```py
.. include:: ../examples/load-dataset-and-visualize.py
```

which generates the figure below:

[<img
    src="https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/visualized.png"
    width=100%
    alt="TaW dataset visualization from genetic algorithm"
    title="TaW"
/>](https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/visualized.png)

Each dataset has a metadata object as well, which is stored in `json` format:

```json
.. include:: ../tce/datasets/tungsten_tantalum_genetic/metadata.json
```

which tells you some info, and (should) additionally give you contact information to inquire about the dataset. Note
that `tce.datasets.Dataset.configurations` is of type `list[ase.Atoms]`, so you can directly plug this into a training
routine. Please contact me directly (email above) if you have datasets you would like to be added 😊

# Sharp Edges

`tce-lib` has a couple of sharp edges (or gotcha's) that one needs to look out for.

## Extensivity of features

In traditional cluster expansion packages, correlation functions are usually intensive (i.e. independent of size). This
is not the case for `tce-lib`. We can showcase this by creating some feature vectors for an FCC lattice of varying
sizes:

```py
.. include:: ../examples/size-dependence.py
```

[<img
    src="https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/size-dependence.png"
    width=100%
    alt="Size dependence of features"
    title="Size dependence"
/>](https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/size-dependence.png)

So, be careful if you are training an intensive property! By default, you will be training intensive properties on
extensive features, which does not make sense. You can fix this by training on an equivalent extensive feature, and
then make the property intensive later, as done in
[the stress example above](https://muexly.github.io/tce-lib/tce.html#learning-a-tensorial-property).

"""

__version__ = "0.3.2"
__authors__ = ["Jacob Jeffries"]

__url__ = "https://github.com/MUEXLY/tce-lib"

import warnings

from . import constants as constants
from . import structures as structures
from . import topology as topology


if __version__.startswith("0."):
    warnings.simplefilter("once", UserWarning)

    warnings.warn(
        f"{__name__} is in alpha. APIs are unstable and may change without notice. "
        f"Please report any problems at {__url__}/issues",
        UserWarning,
        stacklevel=2,
    )
