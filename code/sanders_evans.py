import jax.numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass

class BaseKernel(ABC):
    """A generic base kernel"""
    alpha: float
    beta: float
    gamma: float

    def delta_max(self) -> float:
        return self.alpha/np.sqrt(self.beta + self.gamma)
    
    def get_half_kernel_scales(self, 
                               sigmaK: float, 
                               delta: float) -> tuple[float, float]:
        Delta = sigmaK*delta/self.alpha
        a = np.sqrt((sigmaK**2. - self.gamma*Delta**2.)/self.beta)
        a_minus = a - Delta
        a_plus = a + Delta
        return a_minus, a_plus

    @abstractmethod
    def evaluate_fft(self, 
                     a_minus: float, 
                     a_plus: float, 
                     u: np.array) -> np.array:
        pass

class UniformKernel(BaseKernel):
    
    def __init__(self):
        super().__init__(alpha=0.5, beta=1/3., gamma=1/12.)
    
    def evaluate_fft(self,
                     a_minus: float,
                     a_plus: float,
                     u: np.array) -> np.array:
        # iu = 1j*u
        # iau_min = -a_minus*iu
        # iau_pls = a_plus*iu
        # Fk = 0.5 * ((np.exp(iau_min)-1.)/iau_min + (np.exp(iau_pls)-1.)/iau_pls)
        # # Fk[u==0.] = 1.
        # # Fk = Fk.at[0].set(1.)
        iu = 1j*u[1:]
        iau_min = -a_minus*iu
        iau_pls = a_plus*iu
        Fk = 0.5 * ((np.exp(iau_min)-1.)/iau_min + (np.exp(iau_pls)-1.)/iau_pls)
        Fk = np.insert(Fk, 0, 1.)
        return Fk
    
class CosineKernel(BaseKernel):
    
    def __init__(self):
        super().__init__(
            alpha=1.-2/np.pi, 
            beta=1.-8/np.pi**2, 
            gamma=4/np.pi*(1-3/np.pi)
        )

    def evaluate_fft(self,
                     a_minus: float,
                     a_plus: float,
                     u: np.array) -> np.array:
        iu = 1j*u
        iau_min = -a_minus*iu
        iau_pls = a_plus*iu
        Fk = (np.pi*np.exp(iau_min)-2.*iau_min)/(np.pi**2-4*(a_minus*u)**2)
        Fk = Fk + (np.pi*np.exp(iau_pls)-2.*iau_pls)/(np.pi**2-4*(a_plus*u)**2)
        Fk = Fk * np.pi/2.
        return Fk

class LaplaceKernel(BaseKernel):
    
    def __init__(self):
        super().__init__(alpha=1., beta=2., gamma=1.)
    
    def evaluate_fft(self,
                     a_minus: float,
                     a_plus: float,
                     u: np.array) -> np.array:
        iu = 1j*u
        Fk = 0.5/(1.+a_minus*iu) + 0.5/(1.-a_plus*iu)
        return Fk

class LOSVD(object):
    
    def __init__(self, pkernel_type='uniform', lkernel_type='laplace'):
        self.pkernel_type = pkernel_type
        self.lkernel_type = lkernel_type
        self.set_half_kernels()

    def set_half_kernels(self):
        if self.pkernel_type=='uniform':
            self.pkernel = UniformKernel()
        elif self.pkernel_type=='cosine':
            self.pkernel = CosineKernel()
        else:
            raise ValueError('Unknown platykurtic_kernel')
        if self.lkernel_type=='laplace':
            self.lkernel = LaplaceKernel()
        elif self.lkernel_type=='gaussian':
            self.lkernel = GaussianKernel()
        else:
            raise ValueError('Unknown leptokurtic_kernel')
        self.delta_max = np.min(np.array([
            self.pkernel.delta_max(),
            self.lkernel.delta_max()
        ]))

    def evaluate_kernel_fft(self, u, sigmaK, delta, kappa):
        kappa_weight = (kappa+1)/2.
        pkscales = self.pkernel.get_half_kernel_scales(sigmaK, delta)
        Fkp = self.pkernel.evaluate_fft(*pkscales, u)
        lkscales = self.lkernel.get_half_kernel_scales(sigmaK, delta)
        Fkl = self.lkernel.evaluate_fft(*lkscales, u)
        Fk = (1.-kappa_weight)*Fkp + kappa_weight*Fkl
        return Fk
    
    def evaluate_fft_base(self, u, sigmaK, delta, kappa):
        Fk = self.evaluate_kernel_fft(u, sigmaK, delta, kappa)
        FTN = np.exp(-0.5*u**2)
        return FTN*Fk
        
    def evaluate_fft(self, u, V, sigma, sigmaK, delta, kappa):
        sigma_rescale = sigma/(1.+sigmaK**2)**0.5
        muK = sigmaK*delta
        deltaV = muK*sigma_rescale + V
        fft = self.evaluate_fft_base(u*sigma_rescale, sigmaK, delta, kappa) * np.exp(-1j*deltaV*u)
        return fft

    def evaluate_fft(self, u, V, sigma, sigmaK, delta, kappa):
        sigma_rescale = sigma/(1.+sigmaK**2)**0.5
        muK = sigmaK*delta
        deltaV = muK*sigma_rescale + V
        fft = self.evaluate_fft_base(u*sigma_rescale, sigmaK, delta, kappa) * np.exp(-1j*deltaV*u)
        return fft

    def evaluate_via_fft(self, V, sigma, sigmaK, delta, kappa, vmax=10., nv=401):
        assert nv%2==1
        v = np.linspace(-vmax, vmax, nv)
        dv = v[1]-v[0]
        n = int((nv-1)/2)
        u = np.linspace(0., np.pi, n)/dv
        Flosvd = self.evaluate_fft(u, V, sigma, sigmaK, delta, kappa)
        losvd = np.fft.irfft(Flosvd, v.size)
        losvd = np.roll(losvd, n)/dv
        return v, losvd