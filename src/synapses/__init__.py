"""synapses — representation learning & mechanistic interpretability explorations."""

from synapses.data import binarize, mnist_loaders
from synapses.device import device_report, get_device
from synapses.losses import VAELoss, vae_loss
from synapses.run import RunLogger, set_seed
from synapses.vae import MLPVAE

__all__ = [
    "get_device",
    "device_report",
    "MLPVAE",
    "vae_loss",
    "VAELoss",
    "mnist_loaders",
    "binarize",
    "set_seed",
    "RunLogger",
]
