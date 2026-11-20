"""SIE/EPL parameter conventions shared by automatic fitting stages."""
import math
import numpy as np
import torch


def sie_alpha(coordinates,theta,q,phi,cx,cy):
    """Circularized SIE convention; analytic isothermal deflection."""
    x=coordinates[:,0]-cx;y=coordinates[:,1]-cy
    cp,sp=phi.cos(),phi.sin();xr=cp*x+sp*y;yr=-sp*x+cp*y
    epsilon=(1-q*q).clamp_min(1e-10).sqrt()
    psi=torch.linalg.vector_norm(torch.stack((q*xr,yr)),dim=0).clamp_min(1e-5*q.sqrt())
    scale=theta*q.sqrt()/epsilon
    ax=scale*torch.atan(epsilon*xr/psi)
    ay=scale*torch.atanh((epsilon*yr/psi).clamp(-1+1e-10,1-1e-10))
    return torch.stack((cp*ax-sp*ay,sp*ax+cp*ay),dim=-1)


def vector(macro, slope=False):
    g = macro.get('gamma_ext', 0.)
    angle = 2 * macro.get('phi_ext', 0.)
    v = [math.log(macro['theta_E']), macro['q'], macro['phi'],
         macro['center'][0] / .2, macro['center'][1] / .2,
         g * math.cos(angle) / .2, g * math.sin(angle) / .2]
    return np.array(v + ([macro.get('gamma', 2.)] if slope else []))


def macro_at(v, slope=False, fixed_slope=2.):
    v = np.asarray(v)
    return dict(theta_E=float(np.exp(v[0])), q=float(v[1]),
                phi=float((v[2] + np.pi/2) % np.pi - np.pi/2),
                center=[float(.2*v[3]), float(.2*v[4])],
                gamma_ext=float(.2*np.hypot(v[5], v[6])),
                phi_ext=float(.5*np.arctan2(v[6], v[5])),
                gamma=float(v[7]) if slope else float(fixed_slope))


def macro_bounds(macro, slope=False, wide=False):
    v = vector(macro, slope)
    bounds = [(v[0]+np.log(.7), v[0]+np.log(1.3)), (.25, .995),
              (v[2]-.7, v[2]+.7), (v[3]-.75, v[3]+.75),
              (v[4]-.75, v[4]+.75),
              (max(-2., v[5]-.6), min(2., v[5]+.6)),
              (max(-2., v[6]-.6), min(2., v[6]+.6))]
    if wide:
        bounds=[(v[0]+np.log(.5),v[0]+np.log(1.6)),(.12,.995),(v[2]-np.pi/2,v[2]+np.pi/2),
                (v[3]-1.75,v[3]+1.75),(v[4]-1.75,v[4]+1.75),
                (max(-2.5,v[5]-1.25),min(2.5,v[5]+1.25)),
                (max(-2.5,v[6]-1.25),min(2.5,v[6]+1.25))]
    return bounds + ([(1.6, 2.4)] if slope else [])


def bounded_start(start,bounds):
    """Unwrap the pi-periodic ellipse angle before applying finite bounds."""
    if torch.is_tensor(start):start=start.detach().cpu().numpy()
    start=np.array(start,dtype=float,copy=True)
    limits=np.asarray(bounds)
    midpoint=float(limits[2].mean())
    start[2]+=np.pi*np.rint((midpoint-start[2])/np.pi)
    return np.clip(start,limits[:,0],limits[:,1])
