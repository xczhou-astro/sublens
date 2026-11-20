"""Distributional checks for the experimental positive-source sampler."""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import truncnorm
from scipy.linalg import cholesky,solve_triangular
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from euclid_dr1_joint_lambda import positive_gaussian_sweeps,log_target


def test_positive_source_gibbs_matches_truncated_normal_including_far_tail():
    # Independent dimensions permit checking moments without autocorrelation.
    means=np.array([1.2,-.4,-8.]);precision=np.eye(3)
    draws=[];point=np.ones(3)
    for seed in range(2500):
        point=positive_gaussian_sweeps(precision,means,point,1,seed)
        draws.append(point.copy())
    draws=np.array(draws)
    expected=np.array([truncnorm.mean(-mu,np.inf,loc=mu) for mu in means])
    np.testing.assert_allclose(draws.mean(0),expected,atol=.04)
    assert np.all(draws>=0)


def test_log_lambda_conditional_includes_prior_normalization():
    t=np.array([.2,.8,.4]);r=np.array([[2.,-.3,0],[-.3,1.,-.1],[0,-.1,.7]])
    n=27;u=-1.1;epsilon=1e-6
    numerical=(log_target(8,t,r,n,u+epsilon)-log_target(8,t,r,n,u-epsilon))/(2*epsilon)
    # Derivative with respect to log10 lambda for the Gamma(K/2,rate) conditional.
    expected=np.log(10)*(.5*len(t)-.5*n*10**u*(t@r@t))
    np.testing.assert_allclose(numerical,expected,rtol=1e-8)


def test_gaussian_transport_reverses_and_has_correct_jacobian():
    a=np.array([[4.,1.],[1.,2.] ]);b=np.array([[2.,-.3],[-.3,5.]])
    la=cholesky(a,lower=True);lb=cholesky(b,lower=True)
    ma=np.array([1.,.5]);mb=np.array([.7,.9]);t=np.array([.8,.6])
    transform=solve_triangular(lb.T,la.T,lower=False)
    proposed=mb+transform@(t-ma)
    recovered=ma+solve_triangular(la.T,lb.T@(proposed-mb),lower=False)
    np.testing.assert_allclose(recovered,t,atol=1e-14)
    np.testing.assert_allclose(np.linalg.slogdet(transform)[1],np.log(np.diag(la)).sum()-np.log(np.diag(lb)).sum())
    np.testing.assert_allclose((t-ma)@a@(t-ma),(proposed-mb)@b@(proposed-mb))
