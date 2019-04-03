import numpy as np
import numpy.linalg as npla
import scipy.linalg as scila

def weighted_SVD_basis(curves, error, Neigen=None):
    if type(Neigen) is type(None):
        Neigen = len(error)
    Cinv = np.diag(np.power(error, -2))
    matrix_G = np.dot(curves.T, np.dot(curves, Cinv))
    eigenvalues, eigenvectors = npla.eig(matrix_G)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    argsort = np.argsort(eigenvalues)[-1::-1]
    eigenvectors = eigenvectors[:,argsort]
    eigenvectors = eigenvectors[:,:Neigen]
    norming_matrix = np.dot(np.dot(eigenvectors.T, Cinv), eigenvectors)
    return np.dot(scila.sqrtm(npla.inv(norming_matrix)), eigenvectors.T)
        

def SVD_basis(curves, Neigen=None, return_importances=False):
    """
    Finds a basis from the given set of curves using Singular Value
    Decomposition (SVD).
    
    curves a 2D numpy array of shape (Ncurves, nfreqs)
    Neigen the number of eigenmodes to retain. if Neigen is None, Neigen=nfreqs
    return_importances if True, return the importances of the eigenmodes
                       (default False)
    
    returns numpy.ndarray of shape (Neigen, nfreqs)
            importances of Neigen modes (only if return_importances is True)
    """
    U, importances, basis = npla.svd(curves, full_matrices=True, compute_uv=True)
    if type(Neigen) is not type(None):
        try:
            Neigen = int(Neigen)
        except:
            raise TypeError('Neigen was neither None nor ' +\
                            'an integer-castable type.')
        basis = basis[:Neigen]
        if return_importances:
            importances = importances[:Neigen]
    if return_importances:
        return basis, importances
    else:
        return basis

def SVD_basis_with_chosen_number_of_modes(curves, level, fraction):
    """
    Finds the SVD_basis corresponding to the given curves. Then, chooses the
    number of modes by finding the number necessary for the given fraction of
    the original curves to be fit to within the given level.
    
    curves numpy.ndarray of shape (ncurve, nfreq) filled with training set
    level the desired fitting level
    fraction fraction of curves which are allowed to be fit worse than level
    
    returns numpy.ndarray of shape (N, nfreq) containing the N most important
            SVD basis vectors associated with the given curves where N was
            chosen by this function
    """
    Ncurves = curves.shape[0]
    basis = SVD_basis(curves)
    Neigen = basis.shape[0]
    degrees = np.arange(1, Neigen)
    num_degrees = len(degrees)
    rms = np.ndarray((Ncurves, num_degrees))
    for idegree, degree in enumerate(degrees):
        rms[:,idegree] = SVD_rms_residual(curves, basis[:degree+1])
    fraction_above_level = np.mean((rms > level).astype(int), axis=0)
    fraction_above_level_is_below_required = (fraction_above_level < fraction)
    idegree = np.min(np.where(fraction_above_level_is_below_required)[0])
    return basis[:degrees[idegree]+1,:]



def SVD_coeff(curves, basis, error=None, orthonormal=True):
    """
    Finds the least square fit coefficients which connect the given basis to
    the given curves.
    
    curves numpy.ndarray of shape (Ncurve, Nfreq) or (Nfreq,) if single curve
    basis numpy.ndarray of shape (Neigen, Nfreq) containing basis vectors
    
    returns numpy.ndarray of shape (Neigen, Ncurve) containing coefficients
            for ith curve in result[:,i] or numpy.ndarray of shape (Neigen,)
            if only one curve is given.
    """
    if type(error) is type(None):
        return npla.lstsq(basis.T, curves.T)[0]
    else:
        norm = npla.inv(np.dot(basis / error, (basis / error).T))
        return np.dot(norm, np.dot(basis / error, (curves / error).T))

def SVD_fit(curves, basis, error=None):
    """
    Finds the fits to all curves from the same basis at once. Returns the
    modeled curve.
    
    curves numpy.ndarray of shape (Ncurve, Nfreq) or (Nfreq,) if single curve
    basis numpy.ndarray of shape (Neigen, Nfreq) containing basis vectors
    
    returns numpy.ndarray of shape (Ncurve, Nfreq) or (Nfreq,) if single curve
    """
    return np.dot(SVD_coeff(curves, basis, error=error).T, basis)

def SVD_residual(curves, basis, error=None):
    """
    Finds the residuals of the fits to the given curves given the basis.
    
    curves numpy.ndarray of shape (Ncurve, Nfreq) or (Nfreq,) if single curve
    basis numpy.ndarray of shape (Neigen, Nfreq) containing basis vectors
    
    returns numpy.ndarray of shape (Ncurve, Nfreq) or (Nfreq,) containing fit
            residuals
    """
    return curves - SVD_fit(curves, basis, error=error)

def SVD_rms_residual(curves, basis, error=None):
    """
    Finds the RMS residual when the given basis is used to fit the given
    curve(s).
    
    curves numpy.ndarray of shape (Ncurve, Nfreq) or (Nfreq,) if single curve
    basis numpy.ndarray of shape (Neigen, Nfreq) containing basis vectors
    
    returns either a numpy.ndarray of shape (Ncurve,) or a single float scalar
    """
    if type(error) is type(None):
        return np.sqrt(np.sum(\
            np.power(SVD_residual(curves, basis), 2), axis=-1))
    else:
        return np.sqrt(np.sum(np.power(\
            SVD_residual(curves, basis, error=error) / error, 2), axis=-1))

