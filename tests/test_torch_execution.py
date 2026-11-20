"""Device policy, geometry, positive inversion and full Torch fitting contracts."""
import numpy as np
import pytest
import torch
from scipy.spatial import Delaunay
from scipy.signal import fftconvolve
from scipy.optimize import nnls
from sublens import LensingObservation,torch_backend,resolve_device
from sublens.execution import current_execution
from sublens.inversion.source_plane import PixelSourceProblem
from sublens.inversion.torch_source import TorchDelaunay,TorchSourceProblem,positive_quadratic

DEVICES=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])+(['cuda'] if torch.cuda.is_available() else [])


def observation():
    macro=dict(theta_E=.65,q=.83,phi=.2,gamma=2.,gamma_ext=.025,phi_ext=.4,center=[.015,-.01])
    shape=(22,24);psf=np.array([[.01,.02,.04],[.05,.64,.11],[.03,.06,.04]])
    obs=LensingObservation.from_arrays(np.ones(shape),np.full(shape,.03),psf,pixel_scale=.12,oversample=2,macro_model=macro)
    light=.8*np.exp(-(((obs.bx-.05)/.16)**2+((obs.by+.03)/.21)**2)/2)
    image=fftconvolve(light.mean(-1),psf/psf.sum(),mode='same')
    return LensingObservation.from_arrays(image,np.full(shape,.03),psf,pixel_scale=.12,macro_model=macro,native_rays=obs.image_rays())


def test_device_policy_and_nested_context_restoration():
    assert str(resolve_device('cpu'))=='cpu'
    assert current_execution() is None
    with torch_backend('cpu') as first:
        assert first.dtype=='float64'
        with torch_backend('cpu',dtype='float32') as second:assert current_execution() is second
        assert current_execution() is first
    assert current_execution() is None
    with pytest.raises(ValueError):
        with torch_backend('cpu',source_tolerance=0):pass
    if not torch.cuda.is_available():
        with pytest.raises(ValueError,match='CUDA'):resolve_device('gpu')
    if torch.backends.mps.is_available():
        with pytest.raises(ValueError,match='float64'):
            with torch_backend('mps',dtype='float64'):pass


@pytest.mark.parametrize('device',DEVICES)
def test_torch_triangle_walk_and_barycentric_interpolation(device):
    rng=np.random.default_rng(8);mesh=Delaunay(rng.normal(size=(50,2)))
    points=rng.uniform(-3,3,(1200,2));reference=mesh.find_simplex(points)
    with torch_backend(device) as settings:
        geometry=TorchDelaunay(mesh,settings)
        p=torch.tensor(points,device=device,dtype=settings.torch_dtype)
        simplex,inside=geometry.locate(p)
        np.testing.assert_array_equal(inside.cpu(),reference>=0)
        # Affine sources must reproduce their values independently of triangulation.
        coefficients=2*geometry.points[:,0]-3*geometry.points[:,1]+4
        recovered=geometry.sample(p[:,0,None],p[:,1,None],coefficients).cpu().numpy()
        expected=np.where(reference>=0,2*points[:,0]-3*points[:,1]+4,0.)
        np.testing.assert_allclose(recovered,expected,rtol=2e-6 if device=='mps' else 1e-10,atol=2e-5 if device=='mps' else 1e-10)


@pytest.mark.parametrize('device',DEVICES)
def test_positive_qp_matches_independent_nnls(device):
    rng=np.random.default_rng(13);a=rng.normal(size=(60,18));data=rng.normal(size=60)
    h=a.T@a+np.eye(18)*.1;b=a.T@data
    expected=nnls(np.vstack((a,np.sqrt(.1)*np.eye(18))),np.r_[data,np.zeros(18)])[0]
    with torch_backend(device) as cfg:
        x,steps,error=positive_quadratic(torch.tensor(h,device=device,dtype=cfg.torch_dtype),torch.tensor(b,device=device,dtype=cfg.torch_dtype),None,
            tolerance=cfg.source_tolerance,max_steps=cfg.source_max_steps)
    assert (x>=0).all();assert error<1e-4
    np.testing.assert_allclose(x.cpu(),expected,atol=2e-5 if device=='mps' else 1e-9)


@pytest.mark.parametrize('device',DEVICES)
def test_source_mapping_and_psf_match_reference_without_reference_inner_solve(device,monkeypatch):
    obs=observation();p=PixelSourceProblem(obs.image,obs.rms,obs.psf,obs.mask,obs.bx,obs.by,source_nodes=65,seed=2,regularization=.003)
    expected=p.fit(obs.bx,obs.by)
    with torch_backend(device) as cfg:
        engine=TorchSourceProblem(p,cfg)
        def forbidden(*args,**kwargs):raise AssertionError('Reference mapping/optimizer invoked during Torch solve')
        monkeypatch.setattr(p,'_prepared_operator',forbidden)
        fitted=engine.fit(obs.bx,obs.by)
        result=engine.export(fitted)
        np.testing.assert_allclose(result.image_model,expected.image_model,atol=2e-4)
        assert result.objective<=expected.objective+2e-6
        assert (result.coefficients>=0).all()
        # Source prior changes must not reuse a stale regularization matrix.
        p2=p.with_regularization(.2)
        reweighted=p2.fit(obs.bx,obs.by)
        assert reweighted.regularization==.2


def test_profile_envelope_gradient_matches_reoptimized_difference():
    obs=observation();p=PixelSourceProblem(obs.image,obs.rms,obs.psf,obs.mask,obs.bx,obs.by,source_nodes=65,regularization=.02)
    with torch_backend('cpu') as cfg:
        engine=TorchSourceProblem(p,cfg)
        x=engine.tensor(obs.bx);y=engine.tensor(obs.by)
        shift=torch.tensor(.00123,dtype=torch.float64,requires_grad=True)
        fitted=engine.fit(x+shift,y)
        loss=engine.envelope_loss(x+shift,y,fitted)
        derivative,=torch.autograd.grad(loss,shift)
        step=1e-6
        plus=engine.fit(x+float(shift.detach())+step,y).objective
        minus=engine.fit(x+float(shift.detach())-step,y).objective
        np.testing.assert_allclose(float(derivative),float((plus-minus)/(2*step)),rtol=.003,atol=2e-5)


@pytest.mark.parametrize('device',DEVICES)
def test_pinn_and_matched_joint_refinement_use_saved_backend(device,monkeypatch):
    obs=observation()
    # Neither nonlinear refinement nor source inversion may use scipy.optimize.
    import sublens.inversion.halo_search as halos
    import sublens.inversion.joint_refinement as joint
    import sublens.inversion.source_plane as source
    def forbidden(*args,**kwargs):raise AssertionError('SciPy optimizer used in Torch execution')
    for module in (halos,joint,source):monkeypatch.setattr(module,'minimize',forbidden)
    result=obs.search_pinn(backend='torch',device=device,source_nodes=65,regularization=.003,top_k=1,pinn_steps=2,refine_steps=3)
    assert result.config['execution']['backend']=='torch'
    assert result.config['execution']['device']==device
    assert current_execution() is None
    refined=result.refine_joint(obs.macro_model,maxiter=4,n_starts=1)
    assert current_execution() is None
    assert np.isfinite(refined.baseline.image_model).all()
    for candidate in refined.candidates:
        assert candidate.fit.source.grid is refined.baseline.grid
        assert candidate.fit.delta_objective>=-1e-6


@pytest.mark.parametrize('device',DEVICES)
def test_macro_fit_uses_torch_and_keeps_the_best_profiled_source(device):
    obs=observation()
    with torch_backend(device):
        initial=obs.reconstruct(source_nodes=65,regularization=.003,mesh_strategy='centroidal',seed=42)
        fitted=obs.fit_macro_torch(source_nodes=65,regularization=.003,maxiter=3)
    assert fitted['execution']['device']==device
    assert fitted['source'].objective<=initial.source.objective+2e-6
    assert .2<=fitted['macro_model']['theta_E']<=5.


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware is not available on this runner')
def test_cuda_gpu_alias_allocates_source_fit_on_cuda():
    with torch_backend('gpu') as config:
        assert config.device=='cuda'
        obs=observation()
        fit=obs.reconstruct(source_nodes=45,regularization=.01)
        assert np.isfinite(fit.source.image_model).all()


def test_blank_source_joint_result_has_finite_serializable_diagnostics(tmp_path):
    obs=observation();obs.image[:]=0.
    result=obs.search_pinn(backend='torch',device='cpu',source_nodes=35,top_k=1,pinn_steps=1,refine_steps=1)
    result=result.refine_joint(obs.macro_model,maxiter=2)
    result.save_json(tmp_path/'blank.json')
    assert result.baseline.objective==0.


def test_failed_inner_kkt_is_rejected_before_envelope_gradient(monkeypatch):
    import sublens.inversion.torch_source as module
    obs=observation();problem=PixelSourceProblem(obs.image,obs.rms,obs.psf,obs.mask,obs.bx,obs.by,source_nodes=35)
    with torch_backend('cpu',source_tolerance=1e-7) as cfg:
        engine=TorchSourceProblem(problem,cfg)
        monkeypatch.setattr(module,'positive_quadratic',lambda gram,rhs,initial,**kwargs:(torch.zeros_like(rhs),1,5e-7))
        with pytest.raises(RuntimeError,match='KKT'):engine.fit(obs.bx,obs.by)


def test_explicit_device_selects_torch_and_reference_can_override_context(monkeypatch):
    import sublens.inversion.halo_search as module
    from sublens.execution import selected_execution,result_execution
    obs=observation()
    monkeypatch.setattr(module,'search_pinn_subhalos',lambda *a,**k:'reference' if current_execution() is None else current_execution().device)
    assert obs.search_pinn(device='cpu')=='cpu'
    with torch_backend('cpu',dtype='float32') as settings:
        assert obs.search_pinn(backend='reference')=='reference'
        assert current_execution() is settings
        with selected_execution('reference'):
            with result_execution({'execution':settings.to_dict()}):assert current_execution() is None
    with pytest.raises(ValueError):obs.search_pinn(backend='reference',device='mps')


@pytest.mark.parametrize('device',DEVICES)
def test_simultaneous_halo_context_uses_torch_and_preserves_zero_start(device,monkeypatch):
    from sublens import MultiHaloProblem
    import sublens.inversion.multi_halo as module
    obs=observation()
    model=MultiHaloProblem(obs,source_nodes=45,regularization=.01)
    def forbidden(*args,**kwargs):raise AssertionError('SciPy used in Torch multi-halo fit')
    monkeypatch.setattr(module,'minimize',forbidden)
    start=np.array([0.,0.,0.,.7,.1,0.,-.7,-.1,0.])
    anchors=np.array([[.7,.1],[-.7,-.1]])
    with torch_backend(device):
        initial,_,_=model.profiled_objective(start)
        fitted=model.fit(start,anchors,maxiter=3,n_starts=1,retry_failed=False)
    assert fitted.execution['device']==device
    assert fitted.halos.shape==(2,3)
    assert (fitted.halos[:,2]>=0).all()
    assert fitted.source.objective<=initial+2e-6
