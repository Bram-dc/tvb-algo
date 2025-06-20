import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _cache_folder():
    folder = os.path.expanduser("~/.cache/tvb_algo")
    os.makedirs(folder, exist_ok=True)
    return folder


def rm_cache():
    from shutil import rmtree

    rmtree(_cache_folder())


def tvb76_weights_lengths():
    import zipfile, urllib.request

    cache_fname = os.path.join(_cache_folder(), "tvb76.npz")
    if not os.path.exists(cache_fname):
        logger.info("downloading TVB default connectome")
        with zipfile.ZipFile(
            urllib.request.urlretrieve(
                "https://github.com/the-virtual-brain/tvb-data/"
                "raw/master/tvb_data/connectivity/connectivity_76.zip"
            )[0]
        ) as zf:
            np.savez(
                cache_fname,
                W=np.loadtxt(zf.open("weights.txt")),
                D=np.loadtxt(zf.open("tract_lengths.txt")),
            )
    npz = np.load(cache_fname)

    W = npz["W"]
    D = npz["D"]

    return W, D


def tvb192_weights_lengths():
    import zipfile, urllib.request

    cache_fname = os.path.join(_cache_folder(), "tvb192.npz")
    if not os.path.exists(cache_fname):
        logger.info("downloading TVB default connectome")
        with zipfile.ZipFile(
            urllib.request.urlretrieve(
                "https://github.com/the-virtual-brain/tvb-data/"
                "raw/master/tvb_data/connectivity/connectivity_192.zip"
            )[0]
        ) as zf:
            np.savez(
                cache_fname,
                W=np.loadtxt(zf.open("connectivity_192/weights.txt")),
                D=np.loadtxt(zf.open("connectivity_192/tract_lengths.txt")),
            )
    npz = np.load(cache_fname)

    W = npz["W"]
    D = npz["D"]

    return W, D


def tvb998_weights_lengths():
    import zipfile, urllib.request

    cache_fname = os.path.join(_cache_folder(), "tvb998.npz")
    if not os.path.exists(cache_fname):
        logger.info("downloading TVB default connectome")
        with zipfile.ZipFile(
            urllib.request.urlretrieve(
                "https://github.com/the-virtual-brain/tvb-data/"
                "raw/master/tvb_data/connectivity/connectivity_998.zip"
            )[0]
        ) as zf:
            np.savez(
                cache_fname,
                W=np.loadtxt(zf.open("weights.txt")),
                D=np.loadtxt(zf.open("tract_lengths.txt")),
            )
    npz = np.load(cache_fname)

    W = npz["W"]
    D = npz["D"]

    return W, D
