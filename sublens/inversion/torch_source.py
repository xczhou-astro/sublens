"""Device-resident Delaunay interpolation, exact PSF mapping and positive QP fits.

SciPy constructs fixed topology and a coarse seed table once. No SciPy point
location, convolution or optimization is used in subsequent fitting iterations.
Triangle choices are discrete; barycentric interpolation is differentiable in
ray coordinates within each simplex, as in the reference envelope derivative.
"""
from dataclasses import dataclass
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.fft import next_fast_len
from ..execution import current_execution


class TorchDelaunay:
    def __init__(self,mesh,settings,resolution=128):
        self.device=torch.device(settings.device);self.dtype=settings.torch_dtype
        self.points=torch.as_tensor(mesh.points.copy(),device=self.device,dtype=self.dtype)
        self.simplices=torch.as_tensor(mesh.simplices.copy(),device=self.device,dtype=torch.long)
        self.transforms=torch.as_tensor(mesh.transform.copy(),device=self.device,dtype=self.dtype)
        self.neighbors=torch.as_tensor(mesh.neighbors.copy(),device=self.device,dtype=torch.long)
        lo=mesh.points.min(0);hi=mesh.points.max(0);self.resolution=resolution
        self.lo=torch.as_tensor(lo,device=self.device,dtype=self.dtype)
        self.span=torch.as_tensor(np.maximum(hi-lo,1e-12),device=self.device,dtype=self.dtype)
        axis=(np.arange(resolution)+.5)/resolution
        gx,gy=np.meshgrid(axis,axis);locations=lo+np.column_stack((gx.ravel(),gy.ravel()))*(hi-lo)
        seeds=mesh.find_simplex(locations)
        missing=seeds<0
        if missing.any():
            centers=mesh.points[mesh.simplices].mean(1)
            seeds[missing]=cKDTree(centers).query(locations[missing])[1]
        self.seeds=torch.as_tensor(seeds,device=self.device,dtype=torch.long)

    @torch.no_grad()
    def locate(self,points):
        relative=(points-self.lo)/self.span
        valid_box=((relative>=0)&(relative<=1)).all(1)
        cell=(relative*self.resolution).long().clamp(0,self.resolution-1)
        triangle=self.seeds[cell[:,1]*self.resolution+cell[:,0]].clone()
        pending=valid_box.clone();inside=torch.zeros_like(pending)
        tolerance=2e-6 if self.dtype==torch.float32 else 1e-11
        for _ in range(32):
            ids=torch.nonzero(pending).flatten()
            if ids.numel()==0:break
            transform=self.transforms[triangle[ids]]
            offset=points[ids]-transform[:,2]
            first=torch.einsum('nij,nj->ni',transform[:,:2],offset)
            weights=torch.cat((first,1-first.sum(1,keepdim=True)),1)
            value,opposite=weights.min(1)
            found=value>=-tolerance
            inside[ids[found]]=True;pending[ids[found]]=False
            remaining=ids[~found]
            if remaining.numel():
                neighbor=self.neighbors[triangle[remaining],opposite[~found]]
                escaped=neighbor<0;pending[remaining[escaped]]=False
                triangle[remaining[~escaped]]=neighbor[~escaped]
        # Rare thin-cell/corner cases: exact bounded-memory all-simplex search.
        unresolved=torch.nonzero(pending).flatten()
        chunk=max(1,250000//len(self.transforms))
        for indices in unresolved.split(chunk):
            offset=points[indices,None,:]-self.transforms[None,:,2,:]
            first=torch.einsum('tij,ntj->nti',self.transforms[:,:2],offset)
            weights=torch.cat((first,1-first.sum(2,keepdim=True)),2)
            match=(weights>=-tolerance).all(2)
            any_match=match.any(1)
            # Integer argmax works on MPS, unlike argmax(bool).
            selected=match.to(torch.int32).argmax(1)
            triangle[indices]=selected;inside[indices]=any_match
        return triangle,inside

    def weights(self,bx,by):
        points=torch.stack((bx.reshape(-1),by.reshape(-1)),1)
        triangle,inside=self.locate(points.detach())
        transform=self.transforms[triangle]
        first=torch.einsum('nij,nj->ni',transform[:,:2],points-transform[:,2])
        weights=torch.cat((first,1-first.sum(1,keepdim=True)),1)
        weights=weights.clamp(0,1)*inside[:,None]
        return self.simplices[triangle],weights

    def sample(self,bx,by,coefficients):
        vertices,weights=self.weights(bx,by)
        return (coefficients[vertices]*weights).sum(1).reshape(bx.shape).mean(-1)


@torch.no_grad()
def positive_quadratic(gram,rhs,initial,*,tolerance,max_steps):
    """Block principal pivoting with a projected-gradient fallback and KKT check.

    The quadratic and nonnegative constraint are unchanged. No artificial ridge
    is added to rescue a low-precision factorization.
    """
    scale=gram.diagonal().clamp_min(torch.finfo(gram.dtype).tiny).rsqrt()
    h=scale[:,None]*gram*scale[None,:];b=scale*rhs
    reference=b.abs().max().clamp_min(torch.finfo(h.dtype).eps)
    x=torch.zeros_like(b) if initial is None else (initial/scale).clamp_min(0)
    passive=(b>0) if initial is None else x>0
    best=len(x)+1;patience=3;iterations=0
    for iterations in range(1,min(max_steps,4*len(x))+1):
        indices=torch.nonzero(passive).flatten();x.zero_()
        if indices.numel():
            sub=h[indices][:,indices]
            chol,info=torch.linalg.cholesky_ex(sub)
            if bool((info!=0).any()):break
            x[indices]=torch.cholesky_solve(b[indices,None],chol).squeeze(1)
        gradient=h@x-b
        bad=(passive&(x<-tolerance*reference))|(~passive&(gradient<-tolerance*reference))
        count=int(bad.sum())
        if count==0:
            x=x.clamp_min(0);break
        if count<best:best=count;patience=3
        elif patience>0:patience-=1
        else:
            chosen=torch.nonzero(bad).flatten()[-1]
            passive[chosen]=~passive[chosen];continue
        passive=passive^bad
    x=x.clamp_min(0)
    gradient=h@x-b
    error=torch.where(x>0,gradient,gradient.clamp_max(0)).abs().max()/reference
    if float(error)>tolerance:
        # A safe Lipschitz bound avoids an eigendecomposition on every ray update.
        lipschitz=1.01*h.abs().sum(1).max();z=x.clone();momentum=1.
        for step in range(max_steps):
            nxt=(z-(h@z-b)/lipschitz).clamp_min(0)
            updated=(1+(1+4*momentum**2)**.5)/2
            z=nxt+(momentum-1)/updated*(nxt-x);x=nxt;momentum=updated
            if (step+1)%25==0 or step==max_steps-1:
                gradient=h@x-b
                error=torch.where(x>0,gradient,gradient.clamp_max(0)).abs().max()/reference
                if float(error)<=tolerance:break
        iterations+=step+1
    return scale*x,iterations,float(error)


@dataclass
class TorchSourceFit:
    coefficients:torch.Tensor
    image_model:torch.Tensor
    objective:torch.Tensor
    iterations:int
    optimality:float


class TorchSourceProblem:
    def __init__(self,problem,settings,shared=None):
        if problem.method!='delaunay':raise ValueError('The Torch execution backend currently requires a Delaunay source')
        self.grid=problem.grid;self.regularization=problem.regularization;self.mesh_strategy=problem.mesh_strategy
        self.settings=settings;self.device=torch.device(settings.device);self.dtype=settings.torch_dtype
        self.geometry=shared.geometry if shared is not None else TorchDelaunay(problem.grid,settings)
        self.projection_cache=shared.projection_cache if shared is not None else {}
        self.shape=problem.shape;self.size=problem.size;self.flux_scale=problem.flux_scale
        self.pixel_count=int(np.prod(problem.shape))
        self.mask=torch.as_tensor(problem.mask.copy(),device=self.device,dtype=torch.bool)
        self.weight=torch.as_tensor(problem.weight.copy(),device=self.device,dtype=self.dtype)
        self.target=torch.as_tensor(problem.target.copy(),device=self.device,dtype=self.dtype)
        prior=problem._prior.toarray();prior.flat[::problem.size+1]+=problem._ridge
        self.prior=torch.as_tensor(prior,device=self.device,dtype=self.dtype)
        kernel=problem.convolution_psf
        self.period=tuple(next_fast_len(n+k//2) for n,k in zip(self.shape,kernel.shape))
        embedded=np.zeros(self.period)
        iy=(np.arange(kernel.shape[0])-kernel.shape[0]//2)%self.period[0]
        ix=(np.arange(kernel.shape[1])-kernel.shape[1]//2)%self.period[1]
        embedded[np.ix_(iy,ix)]=kernel
        self.psf_fft=torch.fft.rfft2(self.tensor(embedded))

    def tensor(self,value):
        if torch.is_tensor(value):return value.to(device=self.device,dtype=self.dtype)
        array=np.asarray(value)
        if not array.dtype.isnative:array=array.astype(array.dtype.newbyteorder('='))
        return torch.as_tensor(np.ascontiguousarray(array),device=self.device,dtype=self.dtype)

    def convolve(self,image):
        blurred=torch.fft.irfft2(torch.fft.rfft2(image,s=self.period)*self.psf_fft,s=self.period)
        return blurred[...,:self.shape[0],:self.shape[1]]

    @torch.no_grad()
    def operator(self,bx,by):
        cached=self.projection_cache
        if 'rays' in cached and torch.equal(bx,cached['rays'][0]) and torch.equal(by,cached['rays'][1]):return cached['operator']
        vertices,weights=self.geometry.weights(bx,by)
        pixels=self.pixel_count;samples=bx.shape[-1]
        image_pixel=torch.arange(pixels,device=self.device).repeat_interleave(samples)
        indices=vertices*pixels+image_pixel[:,None]
        dense=torch.zeros(self.size*pixels,device=self.device,dtype=self.dtype)
        dense.scatter_add_(0,indices.flatten(),(weights/samples).flatten())
        native=dense.reshape(self.size,*self.shape)
        operator=torch.empty((int(self.mask.sum()),self.size),device=self.device,dtype=self.dtype)
        for start in range(0,self.size,self.settings.basis_batch_size):
            end=min(self.size,start+self.settings.basis_batch_size)
            operator[:,start:end]=self.convolve(native[start:end])[:,self.mask].T
        operator*=self.flux_scale*self.weight[:,None]
        cached['rays']=(bx.detach().clone(),by.detach().clone());cached['operator']=operator
        cached['gram']=operator.T@operator;cached['rhs']=operator.T@self.target
        return operator

    def render(self,bx,by,coefficients):
        return self.convolve(self.geometry.sample(bx,by,coefficients))

    @torch.no_grad()
    def fit(self,bx,by,initial=None):
        bx=self.tensor(bx);by=self.tensor(by);a=self.operator(bx,by)
        gram=self.projection_cache['gram']+self.prior;rhs=self.projection_cache['rhs']
        if initial is None:initial=self.projection_cache.get('warm_source')
        start=None if initial is None else self.tensor(initial)/self.flux_scale
        normalized,iterations,error=positive_quadratic(gram,rhs,start,tolerance=self.settings.source_tolerance,max_steps=self.settings.source_max_steps)
        # Validate against the original operator residual, rather than subtracting
        # large quadratic constants. Reject failed inner solves before using an envelope gradient.
        residual=a@normalized-self.target
        objective=residual.square().sum()+normalized@(self.prior@normalized)
        allowed=max(1.05*self.settings.source_tolerance,20*torch.finfo(self.dtype).eps)
        if not bool(torch.isfinite(objective)) or error>allowed:
            raise RuntimeError(f'Torch source inversion failed its KKT tolerance: {error:.3g}')
        coefficients=normalized*self.flux_scale
        self.projection_cache['warm_source']=coefficients
        image=self.render(bx,by,coefficients)
        return TorchSourceFit(coefficients,image,objective,iterations,error)

    def envelope_loss(self,bx,by,fit):
        # Source coefficients are at their constrained optimum. Stop gradients
        # through the inner solve, retaining derivatives of rays and PSF render.
        image=self.render(bx,by,fit.coefficients.detach())
        residual=image[self.mask]*self.weight-self.target
        normalized=fit.coefficients.detach()/self.flux_scale
        return residual.square().sum()+normalized@(self.prior@normalized)

    def export(self,fit):
        from .source_plane import PixelSource
        return PixelSource(fit.coefficients.detach().cpu().numpy().astype(float),fit.image_model.detach().cpu().numpy().astype(float),
            'delaunay',self.grid,self.regularization,float(fit.objective),fit.iterations,fit.optimality,self.mesh_strategy)


def torch_problem(problem,settings=None):
    settings=settings or current_execution()
    if settings is None:raise ValueError('Select a Torch execution context')
    key=(settings,problem.regularization)
    cache=getattr(problem,'_torch_execution_cache',None)
    if cache is None:cache={};problem._torch_execution_cache=cache
    if key not in cache:
        shared=next((engine for (configuration,_),engine in cache.items() if configuration==settings),None)
        cache[key]=TorchSourceProblem(problem,settings,shared=shared)
    return cache[key]
