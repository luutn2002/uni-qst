from numpy import pi, exp, linspace, meshgrid
from numpy.random import normal, choice

from skimage.util import random_noise

from qutip.random_objects import rand_dm
from qutip.wigner import qfunc
from qutip import destroy, mesolve, Options, liouvillian

from torch.nn.functional import conv2d
from torch import Tensor, from_numpy, unsqueeze
from torch import float64

from torchvision.transforms import RandomAffine

xvec = linspace(-5, 5, 32)
yvec = linspace(-5, 5, 32)

def gaus2d(x=0, y=0, n0=1):
    return 1. / (pi * n0) * exp(-((x**2 + y**2.0)/n0))

def get_gauss_kernel():
    nth = 3 # Thermal photon number (determines the Gaussian convolution)
    X, Y = meshgrid(xvec, yvec) #get 2D variables instead of 1D
    return gaus2d(X, Y, n0=nth)

def add_photon_noise(rho0, gamma, tlist):
    """
    """
    n = rho0.shape[0]
    a = destroy(n)
    c_ops = [gamma*a,]
    H = -0*(a.dag() + a)
    opts = Options(atol=1e-20, store_states=True, nsteps=1500)
    L = liouvillian(H, c_ops=c_ops)
    states = mesolve(H, rho0, tlist, c_ops=c_ops)

    return states.states

def add_state_noise(dm, sigma=0.01, sparsity=0.01):
    """
    Adds a random density matrices to the input state.
    
    .. math::
        \rho_{mixed} = \sigma \rho_0 + (1 - \sigma)\rho_{rand}$
    Args:
    ----
        dm (`Qobj`): Density matrix of the input pure state
        sigma (float): the mixing parameter specifying the pure state probability
        sparsity (float): the sparsity of the random density matrix
    
    Returns:
    -------
        rho (`numpy arr`): the mixed state density matrix
    """
    hilbertsize = dm.shape[0]
    rho  = (1 - sigma)*dm + sigma*(rand_dm(hilbertsize, sparsity))
    rho = rho/rho.tr()
    return qfunc(rho, xvec, yvec, g=2)

def MixedStateNoise(state, sigma=0.5, sparsity=0.8, to_numpy=False):
    '''
    Add mixed state noise to state.

    Args:
    amount (int): Desired pepper noise amount. Suggested decent amount is =< 0.5.
    '''
    if to_numpy:
        return add_state_noise(state, sigma=sigma, sparsity=sparsity)
    return Tensor(add_state_noise(state, sigma=sigma, sparsity=sparsity))

def GaussianConvolutionTransformation(state, to_numpy=False):
    """
    This is a post measurement op
    Expectation layer that calculates expectation values for a set of operators on a batch of rhos.
    You can specify different sets of operators for each density matrix in the batch.
    Args:
    ----
        state (`Qobj`): state to transform 
    
    Returns:
    -------
        (`Numpy arr/Tensor size (32, 32)`): state with noise
    """
    
    state = conv2d(unsqueeze(from_numpy(qfunc(state, xvec, yvec, g=2)), dim=0), 
                  from_numpy(get_gauss_kernel()[None, None, :, :]).to(float64),  
                  padding='same').squeeze(0)
    
    if to_numpy:
        return state.numpy()
    
    return state
    
def PhotonLossNoise(state, tlist=None, gamma=0.05, to_numpy=False):
    """Add photon loss noise to image.
    Args:
    ----
        state (`Qobj`): state to transform 
        gamma (float): .
    
    Returns:
    -------
        (`Numpy arr/Tensor size (32, 32)`): state with noise
    """
    
    if tlist: 
        photon_loss_states = add_photon_noise(state, gamma, tlist)
    else:
        tlist = linspace(0, 1000, 2000)
        photon_loss_states = add_photon_noise(state, gamma, tlist)
                
        rho_photon_loss = photon_loss_states[555]
        data_photon_loss = qfunc(rho_photon_loss, xvec, yvec, g=2)
            
        if to_numpy: 
            return data_photon_loss
        return Tensor(data_photon_loss).to(dtype=float64)
    
def PepperNoise(state, amount=0.5, to_numpy=False):
    """Add pepper noise to image.

    Args:
        amount (int): Desired pepper noise amount. Suggested decent amount is =< 0.5.
    """
    state = random_noise(qfunc(state, xvec, yvec, g=2), mode="pepper", amount = amount)
    if to_numpy:
        return state
    return Tensor(state).to(dtype=float64)

def AffineTransformation(state, degree=100, shear=5, to_numpy=False):
    """
    """
    state = qfunc(state, xvec, yvec, g=2)
    state = RandomAffine(degree, shear=shear)(unsqueeze(from_numpy(state), dim=0)).squeeze(0)
    if to_numpy:
        return state.numpy()
    return state

def GaussianNoiseTransformation(state, std=0.05, mean=0, to_numpy=False):
    """Add Gaussian noise to image.

    Args:
        std (int): Standard devitation. Suggested is =< 0.2 as in paper.
    """
    state = qfunc(state, xvec, yvec, g=2)
    row,col = state.shape
    state = state + normal(mean,std, size=(row,col))
    
    if to_numpy:
        return state
    return Tensor(state)

    
def RandomNoiseApply(state, to_numpy=False):
    function_list = [MixedStateNoise, GaussianConvolutionTransformation, 
                     PhotonLossNoise, PepperNoise, AffineTransformation, 
                     GaussianNoiseTransformation, None]
    
    func = choice(function_list)
    if func is not None:
        return func(state, to_numpy=to_numpy)
    else:
        state = qfunc(state, xvec, yvec, g=2)
        if to_numpy: return state.full()
        else: return Tensor(state)