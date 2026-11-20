"""Scoped execution settings; statistical models and reference defaults are unchanged."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass,asdict
import torch

_ACTIVE=ContextVar('sublens_torch_execution',default=None)
_REFERENCE=object()


def resolve_device(device='cpu'):
    name=str(device).lower()
    if name=='gpu':name='cuda'
    if name=='auto':
        name='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    resolved=torch.device(name)
    if resolved.type=='cuda':
        if not torch.cuda.is_available():raise ValueError('CUDA GPU requested, but CUDA is unavailable; choose cpu or mps explicitly')
        if resolved.index is not None and resolved.index>=torch.cuda.device_count():raise ValueError('CUDA device index is unavailable')
    elif resolved.type=='mps':
        if not torch.backends.mps.is_available():raise ValueError('Apple MPS requested, but MPS is unavailable')
        if resolved.index not in (None,0):raise ValueError('Only the default Apple MPS device is available')
    elif resolved.type!='cpu':raise ValueError('Supported devices are cpu, gpu/cuda[:index], mps and auto')
    return resolved


@dataclass(frozen=True)
class TorchExecution:
    device:str='cpu'
    dtype:str='float64'
    source_tolerance:float=1e-7
    source_max_steps:int=4000
    basis_batch_size:int=64
    def to_dict(self):return dict(backend='torch',**asdict(self))
    @property
    def torch_dtype(self):return getattr(torch,self.dtype)


def current_execution():
    value=_ACTIVE.get()
    return None if value is _REFERENCE else value


@contextmanager
def torch_backend(device='cpu',*,dtype=None,source_tolerance=None,source_max_steps=4000,basis_batch_size=64):
    """Run source inversion, PINN and physical fits using Torch on one device.

    FITS I/O, mesh topology, initial lookup construction and peak summaries remain
    CPU preprocessing/postprocessing. Ray point-location and fitting loops use
    Torch. gpu aliases CUDA; unavailable explicit devices never fall back silently.
    CPU/CUDA default to float64; MPS supports float32 only.
    """
    resolved=resolve_device(device)
    if isinstance(dtype,torch.dtype):dtype=str(dtype).split('.')[-1]
    dtype=dtype or ('float32' if resolved.type=='mps' else 'float64')
    if dtype not in ('float32','float64'):raise ValueError('Use float32 or float64')
    if resolved.type=='mps' and dtype=='float64':raise ValueError('MPS does not support float64; use float32 or CPU/CUDA float64')
    tolerance=source_tolerance if source_tolerance is not None else (3e-5 if dtype=='float32' else 1e-7)
    if not 0<tolerance<1 or source_max_steps<1 or basis_batch_size<1:raise ValueError('Use positive source tolerance, steps and basis batch size')
    settings=TorchExecution(str(resolved),dtype,float(tolerance),int(source_max_steps),int(basis_batch_size))
    token=_ACTIVE.set(settings)
    try:yield settings
    finally:_ACTIVE.reset(token)


@contextmanager
def result_execution(config):
    """Restore a saved Torch result's backend unless the caller selected one."""
    saved=config.get('execution',{})
    if _ACTIVE.get() is _REFERENCE or current_execution() is not None or saved.get('backend')!='torch':yield current_execution()
    else:
        with torch_backend(**{k:v for k,v in saved.items() if k!='backend'}) as settings:yield settings


@contextmanager
def selected_execution(backend=None,device=None):
    """Explicit devices select Torch; a reference override also blocks auto-restore."""
    if backend is None and device is None:
        yield current_execution()
    elif backend in (None,'torch'):
        if device is None and current_execution() is not None:yield current_execution()
        else:
            with torch_backend(device or 'cpu') as settings:yield settings
    elif backend=='reference':
        if device not in (None,'cpu',torch.device('cpu')):raise ValueError('The reference backend uses CPU; select torch for accelerators')
        token=_ACTIVE.set(_REFERENCE)
        try:yield None
        finally:_ACTIVE.reset(token)
    else:raise ValueError("Choose backend='torch' or backend='reference'")
