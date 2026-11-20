"""Compare exact source objectives and backend timings on one frozen DR1 fit."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time
import numpy as np
import torch
from scipy.optimize import minimize,nnls
from scipy.linalg import cholesky,solve_triangular
sys.path.insert(0,str(Path(__file__).resolve().parent))
from sublens import LensingObservation
from sublens.inversion.source_plane import PixelSourceProblem,_blur_operator,PixelSource
from euclid_dr1_grade_a_scan import write_json,OUT


def quadratic_fit(p,bx,by,initial=None,algorithm='lbfgs'):
    native,a=p._prepared_operator(bx,by)
    h=((a.T@a).toarray() if hasattr(a,'tocsr') else a.T@a)+p._prior.toarray()
    h.flat[::p.size+1]+=p._ridge
    rhs=np.asarray(a.T@p.target).ravel();scale=1/np.sqrt(np.diag(h))
    hs=scale[:,None]*h*scale[None,:];bs=scale*rhs
    start=np.zeros(p.size) if initial is None else initial/(p.flux_scale*scale)
    if algorithm=='lbfgs':
        def fun(x):
            gradient=hs@x-bs
            return float(x@(gradient-bs)+p.target@p.target),2*gradient
        result=minimize(fun,start,jac=True,method='L-BFGS-B',bounds=p._bounds,options=dict(maxiter=1000,gtol=1e-5,ftol=1e-9))
        solution=result.x
    elif algorithm=='nnls':
        l=cholesky(hs,lower=True)
        solution,_=nnls(l.T,solve_triangular(l,bs,lower=True),maxiter=10*p.size)
    elif algorithm=='jax':
        import jax,jax.numpy as jnp
        jax.config.update('jax_enable_x64',True)
        solution,iterations,kkt=fast_jax(hs,bs,start)
        solution=np.asarray(solution)
    source=scale*solution;residual=a@source-p.target
    objective=float(residual@residual+source@(p._prior@source)+p._ridge*(source@source))
    gradient=h@source-rhs
    projected=np.where(source>1e-10,gradient,np.minimum(gradient,0.))
    kkt=float(np.max(np.abs(projected)/np.sqrt(np.diag(h))))
    return objective,p.flux_scale*source,kkt


def fast_jax(h,b,start):
    import jax,jax.numpy as jnp
    return _jax_solver(jnp.asarray(h),jnp.asarray(b),jnp.asarray(start))


def prepare_jax():
    import jax,jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    @jax.jit
    def solver(h,b,start):
        # Safe spectral bound; all iterations execute within one compiled loop.
        lipschitz=jnp.max(jnp.sum(jnp.abs(h),axis=1))*1.01
        reference=jnp.maximum(jnp.max(jnp.abs(b)),1e-14)
        def cond(state):return (state[0]<5000)&(state[4]>1e-7)
        def body(state):
            step,current,z,momentum,error=state
            nxt=jnp.maximum(0.,z-(h@z-b)/lipschitz)
            mt=(1+jnp.sqrt(1+4*momentum*momentum))/2
            extrapolated=nxt+(momentum-1)/mt*(nxt-current)
            gradient=h@nxt-b
            projected=jnp.where(nxt>1e-12,gradient,jnp.minimum(gradient,0.))
            error=jnp.max(jnp.abs(projected))/reference
            return step+1,nxt,extrapolated,mt,error
        steps,point,z,momentum,error=jax.lax.while_loop(cond,body,(0,start,start,1.,jnp.inf))
        return point,steps,error
    return solver


if __name__=='__main__':
    torch.set_num_threads(1)
    obs=LensingObservation.from_fits(OUT/'EUCL-DR1-L020080/observation.fits')
    records=[];reference=None
    for sparse in (False,True):
        p=PixelSourceProblem(obs.image,obs.rms,obs.psf,obs.mask,obs.bx,obs.by,source_nodes=500,
            regularization=2e-6,seed=42,mesh_strategy='centroidal')
        construction=0.
        if sparse:
            now=time.perf_counter();p._blur=_blur_operator(p.shape,p.mask,p.convolution_psf);construction=time.perf_counter()-now
        for algorithm in ('original','lbfgs','nnls','jax','torch_cpu','torch_mps'):
            if algorithm=='jax':
                _jax_solver=prepare_jax()
            trials=[];objectives=[];coefficients=None;kkts=[]
            for index in range(4):
                bx=obs.bx+index*.0002;by=obs.by-index*.00015
                now=time.perf_counter()
                try:
                    if algorithm=='original':
                        fitted=p.fit(bx,by);objective=fitted.objective;coefficients=fitted.coefficients;kkt=None
                    elif algorithm.startswith('torch_'):
                        fitted=p.fit_torch(bx,by,device='mps' if algorithm=='torch_mps' else 'cpu',max_steps=5000,tolerance=1e-7)
                        objective=fitted.objective;coefficients=fitted.coefficients;kkt=fitted.optimality
                    else:objective,coefficients,kkt=quadratic_fit(p,bx,by,algorithm=algorithm)
                    trials.append(time.perf_counter()-now);objectives.append(float(objective));kkts.append(kkt)
                except Exception as e:
                    records.append(dict(sparse=sparse,algorithm=algorithm,error=str(e)));break
            else:
                if not sparse and algorithm=='original':reference=objectives.copy()
                row=dict(sparse=sparse,algorithm=algorithm,construction_seconds=construction,first_seconds=trials[0],
                    median_repeated_seconds=float(np.median(trials[1:])),objectives=objectives,
                    max_objective_difference_from_original=float(np.max(np.abs(np.array(objectives)-reference))),optimality=kkts)
                records.append(row);print(json.dumps(row),flush=True)
            write_json(OUT/'speed_benchmark/source_backends.json',records)
