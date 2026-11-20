"""Torch bounded optimization with source profiling on a single selected device."""
from dataclasses import dataclass,replace
import math
from typing import Optional
import numpy as np
import torch
from ..execution import current_execution
from ..profiles.macro import epl_deflection,external_shear_deflection
from ..profiles.subhalo import truncated_subhalo_deflection
from .source_plane import PixelSourceProblem
from .torch_source import torch_problem


@dataclass
class TorchOptimum:
    values:torch.Tensor
    source:object
    converged:bool
    iterations:int
    message:str
    projected_gradient:float
    relative_objective_change:Optional[float]


def minimize_profiled(engine,rays_at,start,bounds,*,maxiter,initial=None):
    """Finite box constraints via a smooth logit transform and Torch L-BFGS.

    The inner nonnegative source solve is checked before the envelope gradient.
    Its iterations are not unrolled into the outer autograd graph.
    """
    if maxiter<1:raise ValueError('Use a positive iteration limit')
    limits=engine.tensor(bounds);lower=limits[:,0];width=limits[:,1]-lower
    if not bool(torch.isfinite(limits).all()) or bool((width<=0).any()):
        raise ValueError('Use finite ordered box bounds')
    point=engine.tensor(start)
    fraction=((point-lower)/width).clamp(1e-5,1-1e-5)
    z=torch.nn.Parameter(torch.logit(fraction))
    settings=engine.settings;single=settings.dtype=='float32'
    change_tolerance=1e-9 if not single else 1e-7
    optimizer=torch.optim.LBFGS([z],lr=1.,max_iter=maxiter,max_eval=maxiter*3,
        history_size=20,line_search_fn='strong_wolfe',tolerance_grad=3e-6 if not single else 3e-5,
        tolerance_change=change_tolerance)
    # Include the exact supplied point, including zero-strength components on
    # a boundary. A finite logit cannot represent a box endpoint exactly.
    exact_point=point.detach().clamp(lower,lower+width).requires_grad_(True)
    exact_bx,exact_by=rays_at(exact_point)
    exact_fit=engine.fit(exact_bx.detach(),exact_by.detach(),initial=initial)
    exact_loss=engine.envelope_loss(exact_bx,exact_by,exact_fit)
    exact_gradient,=torch.autograd.grad(exact_loss,exact_point)
    relative=(exact_point-lower)/width
    exact_projected=torch.where(((relative<1e-4)&(exact_gradient>0))|((relative>1-1e-4)&(exact_gradient<0)),0.,exact_gradient)
    best=(float(exact_fit.objective),exact_point.detach(),exact_fit,float((exact_projected*width).abs().max()))
    warm=exact_fit.coefficients
    def closure():
        nonlocal best,warm
        optimizer.zero_grad(set_to_none=True)
        values=lower+width*torch.sigmoid(z)
        bx,by=rays_at(values)
        fitted=engine.fit(bx.detach(),by.detach(),initial=warm)
        warm=fitted.coefficients
        loss=engine.envelope_loss(bx,by,fitted)
        if not bool(torch.isfinite(loss)):raise RuntimeError('Torch outer fit has nonfinite loss')
        physical,=torch.autograd.grad(loss,values,retain_graph=True)
        projected=physical.clone();offset=(values-lower)/width
        projected=torch.where(((offset<1e-4)&(projected>0))|((offset>1-1e-4)&(projected<0)),0.,projected)
        error=float((projected*width).abs().max())
        if best is None or float(fitted.objective)<best[0]:
            best=(float(fitted.objective),values.detach().clone(),fitted,error)
        loss.backward();return loss
    optimizer.step(closure)
    state=optimizer.state[z];iterations=int(state.get('n_iter',0))
    # Evaluate the last accepted point as L-BFGS may stop before another closure.
    final_loss=float(closure().detach())
    _,values,source,error=best
    previous=state.get('prev_loss')
    relative_change=(abs(final_loss-previous)/max(1.,abs(final_loss),abs(previous))) if previous is not None else None
    gradient_converged=error<(5e-5 if not single else 5e-4)
    # As in the reference L-BFGS-B fit, convergence may be reached by either
    # a gradient tolerance or relative objective reduction. Requiring only the
    # former incorrectly labels function-tolerance stops as failures and repeats
    # already stable fits. Keep both diagnostics in the result metadata.
    objective_converged=relative_change is not None and relative_change<=change_tolerance
    converged=gradient_converged or objective_converged
    message=('projected gradient tolerance reached' if gradient_converged else
             'relative objective tolerance reached' if objective_converged else
             'iteration budget reached' if iterations>=maxiter else 'line search/stagnation before convergence tolerances')
    return TorchOptimum(values,source,converged,iterations,message,error,relative_change)


def refine_candidate(problem,baseline,rays,initial,bounds,max_strength,maxiter):
    from .halo_search import HaloFit
    engine=torch_problem(problem)
    bx0,by0,x,y=[engine.tensor(a) for a in rays]
    def ray_model(v):
        ax,ay=truncated_subhalo_deflection(x,y,v[0],v[1],v[2]*max_strength)
        return bx0-ax,by0-ay
    outcome=minimize_profiled(engine,ray_model,initial,bounds,maxiter=maxiter,initial=baseline.coefficients)
    count=int(problem.mask.sum());source=engine.export(outcome.source)
    if source.objective>=baseline.objective:
        return HaloFit(float(outcome.values[0]),float(outcome.values[1]),0.,0.,0.,outcome.converged,
            baseline,baseline.image_model,optimizer_message=outcome.message,iterations=outcome.iterations)
    base=np.sum(((baseline.image_model[problem.mask]*problem.weight)-problem.target)**2)
    chi=np.sum(((source.image_model[problem.mask]*problem.weight)-problem.target)**2)
    return HaloFit(float(outcome.values[0]),float(outcome.values[1]),float(outcome.values[2])*max_strength,
        float(count*(base-chi)),float(count*(baseline.objective-source.objective)),outcome.converged,source,source.image_model,
        optimizer_message=outcome.message,iterations=outcome.iterations)


def refine_joint_torch(result,macro_model,*,radius_fraction=.03,shear_offset=.03,maxiter=80,n_starts=1,retry_failed=False):
    from .halo_search import HaloFit
    obs=result.observation;macro=dict(macro_model)
    if not 0<radius_fraction<1 or shear_offset<=0 or maxiter<1 or n_starts<1:raise ValueError('Use valid bounds and positive iteration/start counts')
    settings=current_execution()
    problem=PixelSourceProblem(obs.image,obs.rms,obs.psf,obs.mask,obs.bx,obs.by,
        regularization=result.baseline.regularization,source_grid=result.baseline.grid,
        mesh_strategy=result.baseline.mesh_strategy or 'random')
    engine=torch_problem(problem);x,y=[engine.tensor(a) for a in obs.image_rays()]
    coordinates=torch.stack((x.flatten(),y.flatten()),1)
    theta=float(macro['theta_E']);slope=float(macro.get('gamma',2.));center=macro.get('center',obs.lens_center)
    host,_=epl_deflection(coordinates,theta_E=theta,q=macro.get('q',.8),phi=macro.get('phi',0.),gamma=slope,center=center)
    shear=external_shear_deflection(coordinates,gamma_ext=macro.get('gamma_ext',0.),phi_ext=macro.get('phi_ext',0.),center=center)
    base=torch.stack((engine.tensor(obs.bx),engine.tensor(obs.by)),-1)
    expected=(coordinates-host-shear).reshape(base.shape)
    if not bool(torch.allclose(expected,base,rtol=2e-5,atol=3e-5)):raise ValueError("macro_model must describe observation's input rays")
    host=host.reshape(*x.shape,2);dx,dy=x-center[0],y-center[1]
    shear_x=torch.stack((dx,-dy),-1);shear_y=torch.stack((dy,dx),-1)
    angle=2*float(macro.get('phi_ext',0.));g1=float(macro.get('gamma_ext',0.))*math.cos(angle);g2=float(macro.get('gamma_ext',0.))*math.sin(angle)
    def ray_model(v,halo):
        mapped=base-((1+radius_fraction*v[0])**(slope-1)-1)*host-shear_offset*(v[1]*shear_x+v[2]*shear_y)
        if halo:
            ax,ay=truncated_subhalo_deflection(x,y,v[3],v[4],v[5]*result.max_strength)
            mapped=mapped-torch.stack((ax,ay),-1)
        return mapped[...,0],mapped[...,1]
    logs=[]
    def optimize(start,bounds,initial,halo):
        outcome=minimize_profiled(engine,lambda v:ray_model(v,halo),start,bounds,maxiter=maxiter,initial=initial)
        retried=False
        if retry_failed and not outcome.converged:
            retry=minimize_profiled(engine,lambda v:ray_model(v,halo),outcome.values,bounds,maxiter=2*maxiter,initial=outcome.source.coefficients)
            if float(retry.source.objective)<float(outcome.source.objective)+1e-9:outcome=retry
            retried=True
        v=outcome.values.detach().cpu().numpy();adjusted=dict(macro)
        adjusted['theta_E']=theta*(1+radius_fraction*float(v[0]));s1,s2=g1+shear_offset*v[1],g2+shear_offset*v[2]
        adjusted['gamma_ext']=float(np.hypot(s1,s2));adjusted['phi_ext']=float(.5*np.arctan2(s2,s1))
        logs.append(dict(has_halo=halo,start=np.asarray(start.detach().cpu() if torch.is_tensor(start) else start).tolist(),
            converged=outcome.converged,message=outcome.message,iterations=outcome.iterations,retried=retried,
            objective=float(outcome.source.objective),projected_gradient=outcome.projected_gradient,
            relative_objective_change=outcome.relative_objective_change))
        return v,engine.export(outcome.source),adjusted,outcome.converged
    macro_bounds=[(-1.,1.)]*3
    nv,ns,nm,nok=optimize(np.zeros(3),macro_bounds,result.baseline.coefficients,False)
    null_chi=np.sum(((obs.image[obs.mask]-ns.image_model[obs.mask])/obs.rms[obs.mask])**2)
    candidates=[]
    for candidate in result.candidates:
        fit=candidate.fit;initial=np.r_[np.zeros(3),fit.x,fit.y,fit.strength/result.max_strength]
        half=result.refinement_radius
        bounds=macro_bounds+[(candidate.seed[0]-half,candidate.seed[0]+half),(candidate.seed[1]-half,candidate.seed[1]+half),(0.,1.)]
        starts=[initial];patterns=[(0.,0.,1.),(.04,-.04,1.5),(-.04,.04,.5),(.04,.04,1.)]
        for index in range(n_starts-1):
            trial=initial.copy();sx,sy,scale=patterns[index%len(patterns)]
            if index==0:trial[:3]=nv;trial[3:5]=candidate.seed
            else:trial[3:5]+=[sx,sy]
            trial[5]=max(trial[5]*scale,.015/result.max_strength)
            starts.append(np.clip(trial,np.array(bounds)[:,0],np.array(bounds)[:,1]))
        log_start=len(logs)
        solutions=[optimize(s,bounds,fit.source.coefficients,True) for s in starts]
        minimum=min(s[1].objective for s in solutions)
        selected=min((s for s in solutions if s[1].objective<=minimum+1e-7),key=lambda s:(not s[3],s[1].objective))
        chosen_log=logs[log_start+next(i for i,item in enumerate(solutions) if item is selected)]
        v,source,model,success=selected
        if source.objective>=ns.objective:
            v=np.r_[nv,v[3:5],0.];source,model,success=ns,nm,nok
            chosen_log=logs[0]
        chi=np.sum(((obs.image[obs.mask]-source.image_model[obs.mask])/obs.rms[obs.mask])**2)
        halo=HaloFit(float(v[3]),float(v[4]),float(v[5]*result.max_strength),float(null_chi-chi),
            float(obs.mask.sum()*(ns.objective-source.objective)),success,source,source.image_model,model,
            optimizer_message=chosen_log['message'],iterations=chosen_log['iterations'])
        candidates.append(replace(candidate,fit=halo))
    config={**result.config,'execution':settings.to_dict(),'joint_refinement':dict(macro_model_input=macro,
        radius_fraction=radius_fraction,shear_offset=shear_offset,maxiter=maxiter,n_starts=n_starts,retry_failed=bool(retry_failed),
        optimizer_attempts=logs,free_macro_parameters=['theta_E','shear_1','shear_2'],free_halo_parameters=['x','y','strength'])}
    return replace(result,baseline=ns,candidates=tuple(candidates),macro_model=nm,macro_converged=nok,config=config)


def fit_macro_torch(obs,*,source_nodes=500,regularization=.002,bounds=None,maxiter=80):
    """Seven free EPL parameters (slope fixed), with a positive profiled source.

    The caller supplies an image-derived macro initialization and may specify
    finite bounds. No published or simulated truth is read by this API.
    """
    macro=obs.macro_model
    if macro is None:raise ValueError('Macro fitting requires an initial macro_model')
    problem=PixelSourceProblem(obs.image,obs.rms,obs.psf,obs.mask,obs.bx,obs.by,source_nodes=source_nodes,
        regularization=regularization,mesh_strategy='centroidal',seed=42)
    engine=torch_problem(problem);x,y=[engine.tensor(a) for a in obs.image_rays()]
    coordinates=torch.stack((x.flatten(),y.flatten()),1)
    g1=macro.get('gamma_ext',0.)*math.cos(2*macro.get('phi_ext',0.));g2=macro.get('gamma_ext',0.)*math.sin(2*macro.get('phi_ext',0.))
    initial=np.array([math.log(macro['theta_E']),macro.get('q',.8),macro.get('phi',0.),macro['center'][0]/.2,macro['center'][1]/.2,g1/.2,g2/.2])
    if bounds is None:bounds=[(math.log(.2),math.log(5.)),(.3,.995),(-math.pi,math.pi),(-2.5,2.5),(-2.5,2.5),(-1.5,1.5),(-1.5,1.5)]
    def rays_at(v):
        center=(.2*v[3],.2*v[4]);host,_=epl_deflection(coordinates,theta_E=v[0].exp(),q=v[1],phi=v[2],gamma=macro.get('gamma',2.),center=center)
        dx,dy=coordinates[:,0]-center[0],coordinates[:,1]-center[1]
        shear=.2*torch.stack((v[5]*dx+v[6]*dy,v[6]*dx-v[5]*dy),-1)
        rays=(coordinates-host-shear).reshape(*x.shape,2)
        return rays[...,0],rays[...,1]
    outcome=minimize_profiled(engine,rays_at,initial,bounds,maxiter=maxiter)
    v=outcome.values.detach().cpu().numpy();updated=dict(macro)
    updated.update(theta_E=float(np.exp(v[0])),q=float(v[1]),phi=float((v[2]+math.pi/2)%math.pi-math.pi/2),
        center=[float(.2*v[3]),float(.2*v[4])],gamma_ext=float(.2*np.hypot(v[5],v[6])),phi_ext=float(.5*np.arctan2(v[6],v[5])))
    return dict(macro_model=updated,parameters=v.copy(),source=engine.export(outcome.source),converged=outcome.converged,
        iterations=outcome.iterations,optimizer_message=outcome.message,projected_gradient=outcome.projected_gradient,
        execution=current_execution().to_dict())


@dataclass
class TorchBoxResult:
    x:np.ndarray
    fun:float
    success:bool
    nit:int
    message:str


def minimize_torch_box(loss_function,start,bounds,*,maxiter=200):
    """Torch-only bounded L-BFGS for an already differentiable scalar loss."""
    settings=current_execution()
    if settings is None:raise ValueError('Select torch_backend before calling the Torch box optimizer')
    device=torch.device(settings.device);dtype=settings.torch_dtype
    limits=torch.as_tensor(bounds,device=device,dtype=dtype)
    lower=limits[:,0];width=limits[:,1]-lower
    if not bool(torch.isfinite(limits).all()) or bool((width<=0).any()):raise ValueError('Use finite ordered box bounds')
    first=torch.as_tensor(start,device=device,dtype=dtype)
    parameter=torch.nn.Parameter(torch.logit(((first-lower)/width).clamp(1e-5,1-1e-5)))
    optimizer=torch.optim.LBFGS([parameter],max_iter=maxiter,max_eval=3*maxiter,history_size=20,
        line_search_fn='strong_wolfe',tolerance_grad=2e-6 if dtype==torch.float64 else 3e-5,
        tolerance_change=1e-10 if dtype==torch.float64 else 1e-7)
    best=None
    def closure():
        nonlocal best
        optimizer.zero_grad(set_to_none=True)
        values=lower+width*torch.sigmoid(parameter);loss=loss_function(values)
        if not bool(torch.isfinite(loss)):raise RuntimeError('Torch fit has nonfinite loss')
        gradient,=torch.autograd.grad(loss,values,retain_graph=True)
        relative=(values-lower)/width
        projected=torch.where(((relative<1e-4)&(gradient>0))|((relative>1-1e-4)&(gradient<0)),0.,gradient)
        error=float((projected*width).abs().max())
        if best is None or float(loss.detach())<best[0]:best=float(loss.detach()),values.detach().clone(),error
        loss.backward();return loss
    optimizer.step(closure);closure()
    iterations=int(optimizer.state[parameter].get('n_iter',0));loss,values,error=best
    success=error<(5e-5 if dtype==torch.float64 else 5e-4)
    return TorchBoxResult(values.cpu().numpy().astype(float),loss,success,iterations,
        'projected gradient tolerance reached' if success else 'Torch iteration budget/stagnation before projected gradient tolerance')
