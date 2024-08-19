import jax.numpy as np
import numpy.typing as npt
from scipy import stats, special

class GHLOSVD(object):
    
    def __init__(self, max_order=4):
        self.max_order = max_order
        self.coeffients = self.get_hermite_polynomial_coeffients()

    def get_hermite_polynomial_coeffients(self):
        """Get coeffients for hermite polynomials normalised as in eqn 14 of
        Capellari 2016

        Parameters
        ----------
        max_order : int
            maximum order hermite polynomial desired
            e.g. max_order = 1 --> use h0, h1
            i.e. number of hermite polys = max_order + 1

        Returns
        -------
        array (max_order+1, max_order+1)
            coeffients[i,j] = coef of x^j in polynomial of order i

        """
        coeffients = []
        for i in range(0, self.max_order+1):
            # physicists hermite polynomials
            coef_i = special.hermite(i)
            coef_i = coef_i.coefficients
            # reverse poly1d array so that j'th entry is coeefficient of x^j
            coef_i = coef_i[::-1]
            # scale according to eqn 14 of Capellari 16
            coef_i *= (special.factorial(i) * 2**i)**-0.5
            # fill poly(i) with zeros for 0*x^j for j>i
            coef_i = np.concatenate((coef_i, np.zeros(self.max_order-i)))
            coeffients += [coef_i]
        coeffients = np.vstack(coeffients)
        return coeffients

    def standardise_velocities(self, v, v_mu, v_sig):
        """

        Parameters
        ----------
        v : array
            input velocity array
        v_mu : array (n_regions,)
            gauss hermite v parameters
        v_sig : array (n_regions,)
            gauss hermite sigma parameters

        Returns
        -------
        array (n_regions,) + v.shape
            velocities whitened by array v_mu, v_sigma

        """
        v = np.atleast_2d(v)
        v_mu = np.atleast_1d(v_mu)
        v_sig = np.atleast_1d(v_sig)
        assert v_mu.shape==v_mu.shape
        w = (v.T - v_mu)/v_sig
        w = w.T
        return w

    def evaluate_hermite_polynomials(self,
                                     coeffients,
                                     w,
                                     standardised=True,
                                     v_mu=None,
                                     v_sig=None):
        """

        Parameters
        ----------
        coeffients : array (n_herm, n_herm)
            coefficients of hermite polynomials as given by method
            get_hermite_polynomial_coeffients
        w : array
            if standardised==True
                shape (n_regions, n_vbins), standardised velocities
            else
                shape (n_vbins,), physical velocities
                and arrays v_mu and v_sig with shape (n_regions,) must be set

        Returns
        -------
        array shape (n_hists, n_regions, n_vbins)
            Hermite polynomials evaluated at w in array of

        """
        if not standardised:
            w = self.standardise_velocities(w, v_mu, v_sig)
        result = np.polynomial.polynomial.polyval(w, coeffients.T)
        return result
    
    def evaluate_fft(self, omega, V, sigma, h3, h4):
        """Evaluate analytic Fourier transfrom of Gauss Hermite LOSVD at
        frequencies omega

        Parameters
        ----------
        V : float
            parameter V of the GH-LOSVD
        sigma : float
            parameter sigma of the GH-LOSVD
        h : array-like
            coeffecients of the GH expansion, h[0] = h_0, etc...
        omega : array-like
            frequencies where to evaluate the fourier transform of the GH-LOSVD

        Returns
        -------
        array-like
            The fourier transform of the LOSVD evaluated at omega

        """
        sigma_omega = sigma*omega
        exponent = -1j*omega*V - 0.5*(sigma_omega)**2
        F_gaussian_losvd = np.exp(exponent)
        # jax.polyval expects poly coefficients in descending order, whereas
        # np.polynomial.polynomial.polyval expects descending, so flip it here
        # ALSO FLIP ORDER OF ARGS
        # ALSO do them one by one
        # H_m = []
        # for i in range(self.max_order+1):
        #     H_m += [np.polyval(self.coeffients[i,::-1].T, sigma_omega)]
        # H_m = np.array(H_m)

        H0 = np.ones_like(sigma_omega)
        H3 = np.polyval(self.coeffients[3,::-1].T, sigma_omega)
        H4 = np.polyval(self.coeffients[4,::-1].T, sigma_omega)
        #          m = [0, 1,  2,  3, 4, ...]
        # i_to_the_m = [1, i, -1, -i, 1, ...]
        F_gh_poly = H0 - 1j*h3*H3 + h4*H4
        F_losvd = F_gaussian_losvd * F_gh_poly
        return F_losvd
