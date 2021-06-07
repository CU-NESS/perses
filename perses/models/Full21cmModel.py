"""
Module containing a class representing a full model of the data of 21-cm
measurements made with an arbitrary number of receiver channels. Calling it
produces an output that is equivalent to an array of shape
\\((N_t,N_\\nu,N_c^2)\\) that has been flattened, where \\(N_t\\) is the number
of time bins, \\(N_\\nu\\) is the number of frequency channels, and \\(N_c\\)
is the number of receiver channels. The last axis corresponds to the number of
real correlation quantities that can be formed from pairwise products of the
\\(N_c\\) receiver channel voltages. The first \\(N_c\\) elements of the last
axis are the auto correlations. The remaining \\(N^2-N\\) elements are the
cross-correlations, with each one having real and imaginary components. The
vector form of the last axis is $$\\begin{bmatrix} |V_1|^2 \\\\ \\vdots\
\\\\ |V_{N_c}|^2 \\\\ \\text{Re}(V_1^\\ast V_2) \\\\ \\text{Im}(V_1^\\ast V_2)\
\\\\ \\vdots \\\\ \\text{Re}(V_1^\\ast V_{N_c}) \\\\\
\\text{Im}(V_1^\\ast V_{N_c}) \\\\ \\text{Re}(V_2^\\ast V_3) \\\\\
\\text{Im}(V_2^\\ast V_3) \\\\ \\vdots \\\\  \\text{Re}(V_2^\\ast V_{N_c})\
\\\\ \\text{Im}(V_2^\\ast V_{N_c}) \\\\ \\vdots \\\\\
\\text{Re}(V_{N_c-1}^\\ast V_{N_c}) \\\\ \\text{Im}(V_{N_c-1}^\\ast V_{N_c})\
\\end{bmatrix}$$

**File**: $PERSES/perses/models/Full21cmModel.py  
**Author**: Keith Tauscher  
**Date**: 3 Jun 2021
"""
from __future__ import division
import numpy as np
from pylinex import Model, LoadableModel, load_model_from_hdf5_group

class Full21cmModel(LoadableModel):
    """
    Class representing a full model of the data of 21-cm measurements made with
    an arbitrary number of receiver channels. Calling it produces an output
    that is equivalent to an array of shape \\((N_t,N_\\nu,N_c^2)\\) that has
    been flattened, where \\(N_t\\) is the number of time bins, \\(N_\\nu\\) is
    the number of frequency channels, and \\(N_c\\) is the number of receiver
    channels. The last axis corresponds to the number of real correlation
    quantities that can be formed from pairwise products of the \\(N_c\\)
    receiver channel voltages. The first \\(N_c\\) elements of the last axis
    are the auto correlations. The remaining \\(N^2-N\\) elements are the
    cross-correlations, with each one having real and imaginary components. The
    vector form of the last axis is $$\\begin{bmatrix} |V_1|^2 \\\\ \\vdots\
    \\\\ |V_{N_c}|^2 \\\\ \\text{Re}(V_1^\\ast V_2) \\\\\
    \\text{Im}(V_1^\\ast V_2) \\\\ \\vdots \\\\ \\text{Re}(V_1^\\ast V_{N_c})\
    \\\\ \\text{Im}(V_1^\\ast V_{N_c}) \\\\ \\text{Re}(V_2^\\ast V_3) \\\\\
    \\text{Im}(V_2^\\ast V_3) \\\\ \\vdots \\\\  \\text{Re}(V_2^\\ast V_{N_c})\
    \\\\ \\text{Im}(V_2^\\ast V_{N_c}) \\\\ \\vdots \\\\\
    \\text{Re}(V_{N_c-1}^\\ast V_{N_c}) \\\\\
    \\text{Im}(V_{N_c-1}^\\ast V_{N_c}) \\end{bmatrix}$$
    """
    def __init__(self, gain_models, offset_models, foreground_model,\
        signal_model):
        """
        Initializes a new `Full21cmModel` using the given constituent models.
        In the following \\(N_t\\) is the number of time bins, \\(N_{\\nu}\\)
        is the number of frequencies, and \\(N_c\\) is the number of receiver
        channels.
        
        Parameters
        ----------
        gain_models : sequence
            length \\(N_c\\) list of complex voltage gain
            `pylinex.model.Model.Model` objects that each return 1D
            `numpy.ndarray` objects corresponding to flattened arrays of shape
            \\((N_t,N_{\\nu},2)\\) (the last axis of this shape corresponds to
            the real and imaginary components of the gain)
        offset_models : sequence
            length \\(N_c\\) list of noise temperature
            `pylinex.model.Model.Model` objects that each return 1D
            `numpy.ndarray` objects corresponding to flattened arrays of shape
            \\((N_t,N_{\\nu})\\)
        foreground_model : `pylinex.model.Model.Model`
            a `pylinex.model.Model.Model` object that returns a 1D array
            corresponding to flattened shape \\((N_t,N_{\\nu},N_c^2)\\). The
            last axis corresponds to the real number correlations of the
            channels as in the vector: $$\\begin{bmatrix} |V_1|^2 \\\\ \\vdots\
            \\\\ |V_{N_c}|^2 \\\\ \\text{Re}(V_1^\\ast V_2) \\\\\
            \\text{Im}(V_1^\\ast V_2) \\\\ \\vdots \\\\ \\text{Re}(V_1^\\ast\
            V_{N_c}) \\\\ \\text{Im}(V_1^\\ast V_{N_c}) \\\\\
            \\text{Re}(V_2^\\ast V_3) \\\\ \\text{Im}(V_2^\\ast V_3) \\\\\
            \\vdots \\\\  \\text{Re}(V_2^\\ast V_{N_c}) \\\\\
            \\text{Im}(V_2^\\ast V_{N_c}) \\\\ \\vdots \\\\\
            \\text{Re}(V_{N_c-1}^\\ast V_{N_c}) \\\\\
            \\text{Im}(V_{N_c-1}^\\ast V_{N_c}) \\end{bmatrix}$$
        signal_model : `pylinex.model.Model.Model`
            a `pylinex.model.Model.Model` object that returns a 1D array
            corresponding to flattened shape \\((N_t,N_{\\nu},N_c^2)\\), (see
            `foreground_model` above for what this last axis represents)
        """
        self.gain_models = gain_models
        self.offset_models = offset_models
        self.foreground_model = foreground_model
        self.signal_model = signal_model
    
    @property
    def gain_models(self):
        """
        The \\(N_c\\)-length sequence of complex voltage gain models for each
        receiver channel. Each model should return an array that is equivalent
        to an array of shape \\((N_t,N_{\\nu},2)\\) that has been flattened
        (the last axis represents real and imaginary parts).
        """
        if not hasattr(self, '_gain_models'):
            raise AttributeError("gain_models was referenced before it was " +\
                "set.")
        return self._gain_models
    
    @gain_models.setter
    def gain_models(self, value):
        """
        Setter for `Full21cmModel.gain_models`.
        
        Parameters
        ----------
        value : sequence
            length-\\(N_c\\) list of complex voltage gain
            `pylinex.model.Model.Model` objects that each return 1D
            `numpy.ndarray` objects corresponding to flattened arrays of shape
            \\((N_t,N_{\\nu},2)\\) (the last axis of this shape corresponds to
            the real and imaginary components of the gain)
        """
        if type(value) in sequence_types:
            if all([isinstance(element, Model) for element in value]):
                if all([(element.num_channels == value[0].num_channels)\
                    for element in value]):
                    self._gain_models = value
                else:
                    raise ValueError("Not all gain_models describe the " +\
                        "same number of channels.")
            else:
                raise TypeError("At least one element of gain_models was " +\
                    "not a Model object.")
        else:
            raise TypeError("gain_models was set to a non-sequence.")
    
    @property
    def num_receiver_channels(self):
        """
        The number of receiver channels to correlate, \\(N_c\\)
        """
        if not hasattr(self, '_num_receiver_channels'):
            self._num_receiver_channels = len(self.gain_models)
        return self._num_receiver_channels
    
    @property
    def num_correlations(self):
        """
        The number of receiver correlations, \\(N_c^2\\).
        """
        if not hasattr(self, '_num_correlations'):
            self._num_correlations = (self.num_receiver_channels ** 2)
        return self._num_correlations
    
    @property
    def num_channels_per_correlation(self):
        """
        The number of channels per (real) correlation quantity,
        \\(N_tN_{\\nu}\\).
        """
        if not hasattr(self, '_num_channels_per_correlation'):
            self._num_channels_per_correlation =\
                (self.gain_models[0].num_channels // 2)
        return self._num_channels_per_correlation
    
    @property
    def num_channels(self):
        """
        The total number of channels in the data \\(N_tN_{\\nu}N_c^2\\).
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels =\
                self.num_channels_per_correlation * self.num_correlations
        return self._num_channels
    
    @property
    def offset_models(self):
        """
        The \\(N_c\\)-length sequence of noise temperature models for each
        receiver channel. Each model should return an array that is equivalent
        to an array of shape \\((N_t,N_{\\nu})\\) that has been flattened.
        """
        if not hasattr(self, '_offset_models'):
            raise AttributeError("offset_models was referenced before it " +\
                "was set.")
        return self._offset_models
    
    @offset_models.setter
    def offset_models(self, value):
        """
        Setter for `Full21cmModel.offset_models`.
        
        Parameters
        ----------
        value : sequence
            length-\\(N_c\\) list of noise temperature
            `pylinex.model.Model.Model` objects that each return 1D
            `numpy.ndarray` objects corresponding to flattened arrays of shape
            \\((N_t,N_{\\nu})\\)
        """
        if type(value) in sequence_types:
            if len(value) == self.num_receiver_channels:
                if all([isinstance(element, Model) for element in value]):
                    if all([(element.num_channels ==\
                        self.num_channels_per_correlation)\
                        for element in value]):
                        self._offset_models = value
                    else:
                        raise ValueError("At least one offset_model did " +\
                            "not describe half the number of channels as " +\
                            "the gain_models (because gain models return " +\
                            "real and imaginary components, while offset " +\
                            "models represent real quantities).")
                else:
                    raise TypeError("At least one element of offset_models " +\
                        "was not a Model object.")
            else:
                raise ValueError("offset_models did not have the same " +\
                    "length as gain_models.")
        else:
            raise TypeError("offset_models was set to a non-sequence.")
    
    @property
    def foreground_model(self):
        """
        The `pylinex.model.Model.Model` object that returns the idealized (i.e.
        no receiver biases) form of the beam-weighted foreground in an array
        equivalent to an array of shape \\(N_tN_{\\nu}N_c^2\\) that has been
        flattened.
        """
        if not hasattr(self, '_foreground_model'):
            raise AttributeError("foreground_model was referenced before " +\
                "it was set.")
        return self._foreground_model
    
    @foreground_model.setter
    def foreground_model(self, value):
        """
        Setter for `Full21cmModel.foreground_model`.
        
        Parameters
        ----------
        value : `pylinex.model.Model.Model`
            a `pylinex.model.Model.Model` object that returns a 1D array
            corresponding to flattened shape \\((N_t,N_{\\nu},N_c^2)\\). The
            last axis corresponds to the real number correlations of the
            channels as in the vector: $$\\begin{bmatrix} |V_1|^2 \\\\ \\vdots\
            \\\\ |V_{N_c}|^2 \\\\ \\text{Re}(V_1^\\ast V_2) \\\\\
            \\text{Im}(V_1^\\ast V_2) \\\\ \\vdots \\\\ \\text{Re}(V_1^\\ast\
            V_{N_c}) \\\\ \\text{Im}(V_1^\\ast V_{N_c}) \\\\\
            \\text{Re}(V_2^\\ast V_3) \\\\ \\text{Im}(V_2^\\ast V_3) \\\\\
            \\vdots \\\\  \\text{Re}(V_2^\\ast V_{N_c}) \\\\\
            \\text{Im}(V_2^\\ast V_{N_c}) \\\\ \\vdots \\\\\
            \\text{Re}(V_{N_c-1}^\\ast V_{N_c}) \\\\\
            \\text{Im}(V_{N_c-1}^\\ast V_{N_c}) \\end{bmatrix}$$
        """
        if isinstance(value, Model):
            if value.num_channels == self.num_channels:
                self._foreground_model = value
            else:
                raise ValueError("The foreground model does not describe " +\
                    "the number of channels expected based on gain_models.")
        else:
            raise TypeError("foreground_model was set to a non-Model object.")
    
    @property
    def signal_model(self):
        """
        The `pylinex.model.Model.Model` object that returns the signal's impact
        on the data channel space. It should return an array that is equivalent
        to an array of shape \\((N_t,N_\\nu,N_c^2)\\) that has been flattened.
        """
        if not hasattr(self, '_signal_model'):
            raise AttributeError("signal_model was referenced before it " +\
                "was set.")
        return self._signal_model
    
    @signal_model.setter
    def signal_model(self, value):
        """
        Setter for `Full21cmModel.signal_model`.
        
        Parameters
        ----------
        value : `pylinex.model.Model.Model`
            a `pylinex.model.Model.Model` object that returns a 1D array
            corresponding to flattened shape \\((N_t,N_{\\nu},N_c^2)\\), (see
            `foreground_model` above for what this last axis represents)
        """
        if isinstance(value, Model):
            if value.num_channels == self.num_channels:
                self._signal_model = value
            else:
                raise ValueError("The signal model does not describe the " +\
                    "number of channels expected based on gain_models.")
        else:
            raise TypeError("signal_model was set to a non-Model object.")
    
    @property
    def parameters(self):
        """
        A list of string parameter names. The gain models' parameters are
        first, then offset models' parameters, then foreground and signal
        parameters. The parameter names are the same as the names of the
        parameters of the underlying models with prefixes like "gain0_" or
        "offset2_" or "foreground" or "signal" prepended to them.
        """
        if not hasattr(self, '_parameters'):
            parameters = []
            for (igain, gain_model) in enumerate(self.gain_models):
                parameters.extend(['gain{0:d}_{1!s}'.format(igain, parameter)\
                    for parameter in gain_model.parameters])
            for (ioffset, offset_model) in enumerate(self.offset_models):
                parameters.extend(['offset{0:d}_{1!s}'.format(ioffset,\
                    parameter) for parameter in offset_model.parameters])
            parameters.extend(['foreground_{!s}'.format(parameter)\
                for parameter in self.foreground_model.parameters])
            parameters.extend(['signal_{!s}'.format(parameter)\
                for parameter in self.signal_model.parameters])
            self._parameters = parameters
        return self._parameters
    
    def _make_parameter_slices(self):
        """
        Makes `slice` objects that allow for parameters to be segmented into
        the parameters that will be passed to each model.
        """
        (gain_slices, offset_slices, current_index) = ([], [], 0)
        for gain_model in self.gain_models:
            gain_slices.append(slice(current_index,\
                current_index + gain_model.num_parameters))
            current_index += gain_model.num_parameters
        for offset_model in self.offset_models:
            offset_slices.append(slice(current_index,\
                current_index + offset_model.num_parameters))
            current_index += offset_model.num_parameters
        foreground_slice = slice(current_index,\
            current_index + self.foreground_model.num_parameters)
        current_index += self.foreground_model.num_parameters
        signal_slice = slice(current_index,\
            current_index + self.signal_model.num_parameters)
        current_index += self.signal_model.num_parameters
        self._gain_slices = gain_slices
        self._offset_slices = offset_slices
        self._foreground_slice = foreground_slice
        self._signal_slice = signal_slice
    
    @property
    def gain_slices(self):
        """
        List of `slice` objects that get gain parameters from full parameter
        array.
        """
        if not hasattr(self, '_gain_slices'):
            self._make_parameter_slices()
        return self._gain_slices
    
    @property
    def offset_slices(self):
        """
        List of `slice` objects that get offset parameters from full parameter
        array.
        """
        if not hasattr(self, '_offset_slices'):
            self._make_parameter_slices()
        return self._offset_slices
    
    @property
    def foreground_slice(self):
        """
        `slice` object that gets foreground parameters from full parameter
        array.
        """
        if not hasattr(self, '_foreground_slice'):
            self._make_parameter_slices()
        return self._foreground_slice
    
    @property
    def signal_slice(self):
        """
        `slice` object that gets signal parameters from full parameter
        array.
        """
        if not hasattr(self, '_signal_slice'):
            self._make_parameter_slices()
        return self._signal_slice
    
    def make_gain_matrix(self, pars):
        """
        Makes the gain matrix from the `pars`, `Full21cmModel.gain_slices`, and
        `Full21cmModel.gain_models`.
        
        Parameters
        ----------
        pars : `numpy.ndarray`
            either a full parameter array or simply a gain parameter array,
            with the values of the parameters of the gain models concatenated
        
        Returns
        -------
        gain_matrix : `distpy.util.SparseSquareBlockDiagonalMatrix.SparseSquareBlockDiagonalMatrix`
            a block diagonal matrix, whose blocks (corresponding to the
            correlations) are themselves block diagonal. Multiplying a vector
            of correlations by this matrix produces the following effects:
            
            1. The \\(k^{\\text{th}}\\) autocorrelation is scaled by
            \\(|g_k|^2\\)
            2. The complex number \\(V^\\ast_m V_n\\) is multiplied by
            \\(g^\\ast_mg_n\\) (meaning the real and imaginary parts of
            \\(V^\\ast_mV_n\\) are scaled by \\(|g^\\ast_mg_n|\\) and then
            rotated by an angle \\(\\text{arg}(g^\\ast_mg_n)\\))
        """
        gains = [gain_model(pars[gain_slice]) for (gain_model,\
            gain_slice) in zip(self.gain_models, self.gain_slices)]
        gains = [np.reshape(gain, (-1, 2)) for gain in gains]
        gains = [(((1. + 0.j) * gain[:,0]) + ((0. + 1.j) * gain[:,1]))\
            for gain in gains]
        gain_blocks = np.zeros((self.num_channels_per_correlation,) +\
            ((self.num_correlations,) * 2))
        for igain in range(self.num_receiver_channels):
            gain_conj = np.conj(gains[igain])
            for jgain in range(igain, self.num_receiver_channels):
                relevant_product = gain_conj * gains[jgain]
                if jgain == igain:
                    gain_blocks[:,igain,igain] = np.real(relevant_product)
                    real_index =\
                        ((1 + (2 * igain)) * self.num_receiver_channels) -\
                        (igain * (igain +  1)) - 2
                else:
                    real_index += 2
                    imag_index = real_index + 1
                    real_part = np.real(relevant_product)
                    imag_part = np.imag(relevant_product)
                    gain_blocks[:,real_index,real_index] = real_part
                    gain_blocks[:,imag_index,imag_index] = real_part
                    gain_blocks[:,real_index,imag_index] = -imag_part
                    gain_blocks[:,imag_index,real_index] = imag_part
        return SparseSquareBlockDiagonalMatrix(gain_blocks)
    
    def __call__(self, pars):
        """
        Gets the full, noiseless data auto- and cross-correlations by
        evaluating this model at the given parameters.
        
        Parameters
        ----------
        pars : `numpy.ndarray`
            array of parameters of each submodel, concatenated, as in the
            following vector: $$\\begin{bmatrix} \\boldsymbol{x}_{g_1} \\\\\
            \\boldsymbol{x}_{g_2} \\\\ \\vdots \\\\\
            \\boldsymbol{x}_{g_{N_c}} \\\\ \\boldsymbol{x}_{o_1} \\\\\
            \\boldsymbol{x}_{o_2} \\\\ \\vdots \\\\\
            \\boldsymbol{x}_{o_{N_c}} \\\\ \\boldsymbol{x}_f \\\\\
            \\boldsymbol{x}_s \\end{bmatrix}$$
        
        Returns
        -------
        data : `numpy.ndarray`
            an array that contains all \\(N_tN_\\nu N_c^2\\) data channels in
            an array similar to the following vector: $$\\begin{bmatrix}\
            |V_1|^2 \\\\ \\vdots \\\\ |V_{N_c}|^2 \\\\\
            \\text{Re}(V_1^\\ast V_2) \\\\ \\text{Im}(V_1^\\ast V_2) \\\\\
            \\vdots \\\\ \\text{Re}(V_1^\\ast V_{N_c}) \\\\\
            \\text{Im}(V_1^\\ast V_{N_c}) \\\\ \\text{Re}(V_2^\\ast V_3) \\\\\
            \\text{Im}(V_2^\\ast V_3) \\\\ \\vdots \\\\\
            \\text{Re}(V_2^\\ast V_{N_c}) \\\\\
            \\text{Im}(V_2^\\ast V_{N_c}) \\\\ \\vdots \\\\\
            \\text{Re}(V_{N_c-1}^\\ast V_{N_c}) \\\\\
            \\text{Im}(V_{N_c-1}^\\ast V_{N_c}) \\end{bmatrix}$$
        """
        gain_matrix = self.make_gain_matrix(pars)
        offset = np.stack([offset_model(pars[offset_slice]) for (offset_model,\
            offset_slice) in zip(self.offset_models, self.offset_slices)],\
            axis=-1)
        offset = np.concatenate([offset, np.zeros((offset.shape[0],\
            self.num_receiver_channels * (self.num_receiver_channels - 1)))],\
            axis=-1).flatten()
        foreground = self.foreground_model(pars[self.foreground_slice])
        signal = self.signal_model(pars[self.signal_slice])
        return gain_matrix.__matmul__(foreground + signal + offset)
    
    @property
    def gradient_computable(self):
        """
        `False`, indicating that derivatives of the model cannot be
        analytically evaluated.
        """
        return False
    
    @property
    def hessian_computable(self):
        """
        `False`, indicating that derivatives of the model cannot be
        analytically evaluated.
        """
        return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model so it
        can be loaded later.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'Full21cmModel'
        group.attrs['import_string'] =\
            'from perses.models import Full21cmModel'
        group.attrs['num_receiver_channels'] = self.num_receiver_channels
        subgroup = group.create_group('gains')
        for (igain, gain_model) in enumerate(self.gain_models):
            gain_model.fill_hdf5_group(\
                subgroup.create_group('gain_{:d}'.format(igain)))
        subgroup = group.create_group('offsets')
        for (ioffset, offset_model) in enumerate(self.offset_models):
            offset_model.fill_hdf5_group(\
                subgroup.create_group('offset_{:d}'.format(ioffset)))
        self.foreground_model.fill_hdf5_group(group.create_group('foreground'))
        self.signal_model.fill_hdf5_group(group.create_group('signal'))
    
    def __eq__(self, other):
        """
        Checks for equality with another object.
        
        Parameters
        ----------
        other : object
            object to check for equality
        
        Returns
        -------
        result : bool
            True if and only if `other` is a `Full21cmModel` with the same
            `Full21cmModel.gain_models`, `Full21cmModel.offset_models`,
            `Full21cmModel.foreground_model`, and `Full21cmModel.signal_model`
        """
        if isinstance(other, Full21cmModel):
            return (self.gain_models == other.gain_models) and\
                (self.offset_models == other.offset_models) and\
                (self.foreground_model == other.foreground_model) and\
                (self.signal_model == other.signal_model)
        else:
            return False
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a `Full21cmModel` from the given hdf5 file group, where it was
        once saved.
        
        Parameters
        ----------
        group : h5py.Group
            hdf5 file group where a `Full21cmModel` was once saved
        
        Returns
        -------
        loaded_model : `Full21cmModel`
            the model that was once saved in `group`
        """
        num_receiver_channels = group.attrs['num_receiver_channels']
        foreground_model = load_model_from_hdf5_group('foreground')
        signal_model = load_model_from_hdf5_group('signal')
        (gain_models, offset_models) = ([], [])
        for index in range(num_receiver_channels):
            gain_models.append(load_model_from_hdf5_group(\
                group['gains/gain_{:d}'.format(index)]))
            offset_models.append(load_model_from_hdf5_group(\
                group['offsets/offset_{:d}'.format(index)]))
        return Full21cmModel(gain_models, offset_models, foreground_model,\
            signal_model)
    
    @property
    def bounds(self):
        """
        The bounds of the parameters, taken from the bounds of the submodels.
        """
        if not hasattr(self, '_bounds'):
            bounds = {}
            for (igain, gain_model) in enumerate(self.gain_models):
                for parameter in gain_model.parameters:
                    bounds['gain{0:d}_{1!s}'.format(igain, parameter)] =\
                        gain_model.bounds[parameter]
            for (ioffset, offset_model) in enumerate(self.offset_models):
                for parameter in offset_model.parameters:
                    bounds['offset{0:d}_{1!s}'.format(ioffset, parameter)] =\
                        offset_model.bounds[parameter]
            for parameter in self.foreground_model.parameters:
                bounds['foreground_{!s}'.format(parameter)] =\
                    self.foreground_model.bounds[parameter]
            for parameter in self.signal_model.parameters:
                bounds['signal_{!s}'.format(parameter)] =\
                    self.signal_model.bounds[parameter]
            self._bounds = bounds
        return self._bounds

