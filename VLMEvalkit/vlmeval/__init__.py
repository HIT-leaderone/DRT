import os
import os.path as osp
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# Temporarily bypass SSL certificate verification to download files from oss.


def _ensure_writable_dir(env_name, preferred, fallback):
    current = os.environ.get(env_name)
    candidates = [current, preferred, fallback]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            os.makedirs(candidate, exist_ok=True)
            if os.access(candidate, os.W_OK):
                os.environ[env_name] = candidate
                return
        except OSError:
            continue


home = osp.expanduser('~')
_ensure_writable_dir('LMUData', osp.join(home, 'LMUData'), '/tmp/LMUData')
_ensure_writable_dir('HF_HOME', osp.join(home, '.cache', 'huggingface'), '/tmp/huggingface')
_ensure_writable_dir('HUGGINGFACE_HUB_CACHE', osp.join(os.environ['HF_HOME'], 'hub'), '/tmp/huggingface/hub')
_ensure_writable_dir('MPLCONFIGDIR', osp.join(home, '.config', 'matplotlib'), '/tmp/matplotlib')

try:
    import torch
except ImportError:
    pass

from .smp import *
load_env()

from .api import *
from .dataset import *
from .utils import *
from .vlm import *
from .config import *
from .tools import cli


__version__ = '0.2rc1'
