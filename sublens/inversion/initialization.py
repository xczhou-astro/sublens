"""Image-only four-start SIE/shear/Sersic initialization, independent of survey.

Source Sersic light is initialization only. No released mass or source products
are read. Angular bounds and the centering prior are recorded with the fit.
"""
import math
from time import perf_counter
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from scipy.fft import next_fast_len
from ..execution import current_execution
from .macro_parameters import sie_alpha, macro_at


def radial_start(obs):
    x,y=obs.image_rays();x=x.mean(-1);y=y.mean(-1)
    radius=np.hypot(x,y);maximum=min(4.,.85*max(radius[obs.mask]))
    bins=np.arange(0,maximum+.05,.05);index=np.minimum(np.digitize(radius,bins),len(bins)-1)
    valid=obs.mask&(radius>.3)&(radius<maximum)
    weights=np.maximum(obs.image/obs.rms-3.,0.)
    sums=np.bincount(index[valid],weights=weights[valid],minlength=len(bins))
    counts=np.bincount(index[valid],minlength=len(bins))
    profile=gaussian_filter1d(sums/np.maximum(counts,1),1.)
    profile[bins<.35]=0
    guess=float(bins[int(np.argmax(profile))])
    return np.clip(guess,.4,3.5),dict(radial_peak_arcsec=guess,radial_bins_arcsec=bins.tolist(),radial_snr_profile=profile.tolist())


class ParametricInitializer:
    def __init__(self,obs):
        self.obs=obs
        settings=current_execution()
        self.device=torch.device(settings.device if settings is not None else 'cpu')
        self.dtype=settings.torch_dtype if settings is not None else torch.float64
        x,y=obs.image_rays();self.shape=x.shape
        self.coordinates=torch.tensor(np.column_stack((x.ravel(),y.ravel())),dtype=self.dtype,device=self.device)
        self.mask=torch.as_tensor(obs.mask,device=self.device)
        self.noise=torch.as_tensor(obs.rms[obs.mask],dtype=self.dtype,device=self.device)
        self.data=torch.as_tensor(obs.image[obs.mask],dtype=self.dtype,device=self.device)
        self.period=tuple(next_fast_len(n+p//2) for n,p in zip(obs.image.shape,obs.psf.shape))
        psf=np.zeros(self.period)
        iy=(np.arange(obs.psf.shape[0])-obs.psf.shape[0]//2)%self.period[0]
        ix=(np.arange(obs.psf.shape[1])-obs.psf.shape[1]//2)%self.period[1]
        psf[np.ix_(iy,ix)]=obs.psf
        self.frequency=torch.fft.rfft2(torch.as_tensor(psf,dtype=self.dtype,device=self.device))
        self.bounds=[(np.log(.2),np.log(5.)),(.3,.995),(-np.pi,np.pi),(-1.,1.),(-1.,1.),(-1.5,1.5),(-1.5,1.5),
            (-1.5,1.5),(-1.5,1.5),(np.log(.015),np.log(1.5)),(.25,.995),(-np.pi,np.pi),(.5,5.)]

    def render(self,v):
        cx,cy=.2*v[3],.2*v[4]
        host=sie_alpha(self.coordinates,v[0].exp(),v[1],v[2],cx,cy)
        x=self.coordinates[:,0]-cx;y=self.coordinates[:,1]-cy;g1,g2=.2*v[5],.2*v[6]
        shear=torch.stack((g1*x+g2*y,g2*x-g1*y),dim=-1)
        beta=self.coordinates-host-shear
        dx,dy=beta[:,0]-v[7],beta[:,1]-v[8]
        cp,sp=v[11].cos(),v[11].sin();xr=cp*dx+sp*dy;yr=-sp*dx+cp*dy
        rr=(v[10]*xr*xr+yr*yr/v[10]+1e-10).sqrt();n=v[12]
        bn=2*n-1/3+4/(405*n)
        brightness=torch.exp((-bn*((rr/v[9].exp()).pow(1/n)-1)).clamp(-100,30))
        native=brightness.reshape(self.shape).mean(-1)
        blurred=torch.fft.irfft2(torch.fft.rfft2(native,s=self.period)*self.frequency,s=self.period)[:self.obs.image.shape[0],:self.obs.image.shape[1]]
        model=blurred[self.mask]/self.noise;target=self.data/self.noise
        amplitude=(model@target/(model@model).clamp_min(1e-30)).clamp_min(0)
        return amplitude*blurred,amplitude

    def loss_tensor(self,v):
        model,amplitude=self.render(v)
        residual=(model[self.mask]-self.data)/self.noise
        # Declared weak mass/light centering prior, sigma=0.15 arcsec; no released mass prior.
        return (residual@residual+(.2*v[3]/.15)**2+(.2*v[4]/.15)**2)/self.obs.mask.sum()

    def objective(self,values):
        v=torch.tensor(values,dtype=self.dtype,device=self.device,requires_grad=True)
        loss=self.loss_tensor(v)
        gradient,=torch.autograd.grad(loss,v)
        return float(loss.detach()),gradient.detach().cpu().numpy()

    def fit(self,maxiter=160):
        start_time=perf_counter();radius,radial=radial_start(self.obs);records=[];states=[]
        for k,(factor,q,phi) in enumerate(((1.,.85,0.),(.8,.75,.7),(1.2,.9,-.7),(1.,.7,1.4))):
            initial=np.array([np.log(radius*factor),q,phi,0.,0.,0.,0.,.02,-.02,np.log(.2),.8,0.,1.])
            initial=np.clip(initial,np.array(self.bounds)[:,0],np.array(self.bounds)[:,1])
            if current_execution() is not None:
                from sublens.inversion.torch_refinement import minimize_torch_box
                result=minimize_torch_box(self.loss_tensor,initial,self.bounds,maxiter=maxiter)
            else:
                result=minimize(self.objective,initial,jac=True,bounds=self.bounds,method='L-BFGS-B',
                    options=dict(maxiter=maxiter,maxls=30,ftol=1e-10,gtol=2e-6))
            records.append(dict(start=initial.tolist(),objective=float(result.fun),converged=bool(result.success),
                iterations=int(result.nit),message=str(result.message),macro=macro_at(result.x[:7])))
            states.append(result.x.copy())
        best=int(np.argmin([r['objective'] for r in records]));v=states[best]
        with torch.no_grad():image,amplitude=self.render(torch.as_tensor(v,dtype=self.dtype,device=self.device))
        return dict(macro=records[best]['macro'],converged=records[best]['converged'],parameters=v.tolist(),
            amplitude=float(amplitude),image_model=image.cpu().numpy(),objective=records[best]['objective'],
            attempts=records,radial=radial,seconds=perf_counter()-start_time,published_mass_used=False,
            weak_center_prior_sigma_arcsec=.15,point_estimate_not_posterior=True)
