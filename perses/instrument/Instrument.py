import numpy as np
from types import FunctionType
from ..util import ParameterFile

class SimpleInstrument(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)
        
        # For convenient access
        self.band = self.pf['instr_band']
        self.channel = self.pf['instr_channel']
        self.Tinstr = self.pf['instr_temp']
        
        if type(self.pf['instr_response']) is FunctionType:
            self.resp = self.pf['instr_response']
        else:
            self.resp = lambda nu: self.pf['instr_response']
        
    def response(self, nu):
        return self.resp(nu)

class SMatrixElement:
  """Stores the polynomial coefficients defining an element of the S matrix.
  The first (second) column stores the real (imaginary) part."""
  def __init__(self,coeffs):
    assert coeffs.shape[1] == 2
    self.coeffs = coeffs
    self.order = coeffs.shape[0] - 1

class AntReflecCoeff:
  """Stores parameters of the antenna reflection coefficient. Remember this
  is in general complex! If using inputtype=1, the real and imaginary parts
  are stored separately, so that c and p are arrays with second dimension of
  length 2. If using inputtype=2, the list of response values may be complex."""
  def __init__(self,inputtype=1,parlist=[0,np.array([[0.387,0.0],[0.0,0.0]]),
                                         np.array([0,0]).T]):
    self.inputtype = inputtype
    if inputtype == 1: # Parametrized in terms of w, c and p
      self.w = parlist[0]
      self.c = parlist[1]
      assert self.c.shape[1] == 2
      self.order = self.c.shape[0] - 1
      self.p = parlist[2]
      assert self.p.shape[1] == 2
      assert self.p.shape[0] == self.order
    elif inputtype == 2: # Just a list of reflection coefficient vs. frequency
      self.nutable = parlist[0]
      self.resptable = parlist[1]
    else:
      raise NotImplementedError(
        'inputtype must be 1 or 2 in AntReflecCoeff.__init__')
  def calc_resp(self,outfreq):
    if self.inputtype == 1:
      """Computes the reflection coefficient of the antenna using the
      parametrization given in Rich Bradley's calibration memo."""
      ga_re = np.ones(outfreq.size) * self.c[0,0]
      ga_im = np.ones(outfreq.size) * self.c[0,1]
      for i in range(self.p[:,0].size):
        ga_re = (ga_re + self.c[i+1,0] * np.sin((i + 1) 
                 * self.w * outfreq + self.p[i,0]))
        ga_im = (ga_im + self.c[i+1,1] * np.sin((i + 1)
                 * self.w * outfreq + self.p[i,1]))
      return ga_re + 1j * ga_im
    if self.inputtype == 2:
      """Computes the reflection coefficient of the antenna using a tabulated
      response function."""
      ga_re = np.interp(outfreq,self.nutable,self.resptable.real)
      ga_im = np.interp(outfreq,self.nutable,self.resptable.imag)
      return ga_re + 1j * ga_im

class NoiseParameters:
  """Stores noise parameters of the amplifier."""
  def __init__(self,Ropt,Xopt,Tmin,Rn,Zzero,Tref,Tamb=300.0):
    self.order = Ropt.size - 1
    assert Xopt.size == Tmin.size == Rn.size == self.order + 1
    self.Ropt = Ropt
    self.Xopt = Xopt
    self.Tmin = Tmin
    self.Rn = Rn
    self.Zzero = Zzero
    self.Tref = Tref
    self.Tamb = Tamb
    self.expanded = False
  def expand(self,outfreq):
    if not self.expanded:
      self.Ropt = np.polyval(self.Ropt,outfreq)
      self.Xopt = np.polyval(self.Xopt,outfreq)
      self.Tmin = np.polyval(self.Tmin,outfreq)
      self.Rn = np.polyval(self.Rn,outfreq)
      self.expanded = True

class Instrument:
  """Stores the parameters of the antenna, amplifier and receiver, allowing the
  instrument response to be computed.

  If inputtype=1, use the description of the instrument from the first
  version of Rich's calibration memo. In particular, note that what
  will be computed is 'T_modelled', which isn't really an easy-
  to-understand spectrum in any sense.

  If inputtype=2, just model the instrument as producing a spectrum
  T_Ant=G_ant*T_sky + T_Rx , ignoring any overall gain, which would
  cancel when computing the constraints if perfectly known anyway.
  G_ant is just 1-|Gamma_ant|^2 in this simple case.

  If inputtype=3, the 'modelled' spectrum is identical to the sky
  spectrum, i.e. perfect calibration is assumed. The noise, however,
  is modelled taking into account contributions from the antenna,
  receiver, and thermal noise of the load (which requires having the
  ambient temperature as a parameter), as laid out in Abhi's memo.

  """
  def __init__(self,inputtype=1,**kwargs):
    self.inputtype = inputtype
    if inputtype == 1 or inputtype == 3:
      self.gamma_ant = kwargs['gamma_ant']
      assert isinstance(self.gamma_ant,AntReflecCoeff)
      self.noisepars = kwargs['noisepars']
      assert isinstance(self.noisepars,NoiseParameters)
      self.S = kwargs['S']
      for i in [0,1]:
        for j in [0,1]:
          assert isinstance(self.S[i][j],SMatrixElement)
      self.T_Rx = kwargs['T_Rx']
      self.gamma_L = kwargs['gamma_L']
    elif inputtype == 2:
      self.gamma_ant = kwargs['gamma_ant']
      assert isinstance(self.gamma_ant,AntReflecCoeff)
      self.T_Rx = kwargs['T_Rx']
    else:
      raise NotImplementedError('inputtype must be 1, 2 or 3 in Instrument')
  def process_skyt(self,sky_temp,auxparams,precomputed={},return_more=False):
    if self.inputtype == 1:
      if 'gamma_ant' in precomputed:
        g_a = precomputed['gamma_ant']
      else:
        g_a = self.gamma_ant.calc_resp(auxparams.outfreq)
      if self.noisepars.expanded == False:
        self.noisepars.expand(auxparams.outfreq)
      Z_ant = 0.5 * self.noisepars.Zzero * (1 + g_a) / (1 - g_a)
      T_amp1 = calc_Tamp(Z_ant,self.noisepars,auxparams.outfreq)
      if 'S' in precomputed:
        S = precomputed['S']
      else:
        S = calc_S(self.S,auxparams.outfreq)
      gamma_L = calc_gamma_L(self.gamma_L,auxparams.outfreq)
      G_amp0 = calc_Gamp0(S[:,1,0],S[:,1,1],gamma_L)
      T_Rx = calc_TRx(self.T_Rx,auxparams.outfreq)
      g = calc_gainratio(S,g_a,gamma_L)
      Tmod = [g * ( oneT + T_amp1) + T_Rx / G_amp0 for oneT in sky_temp]
      Tsys = Tmod
    elif self.inputtype == 2:
      if 'gamma_ant' in precomputed:
        g_a = precomputed['gamma_ant']
      else:
        g_a = self.gamma_ant.calc_resp(auxparams.outfreq)
      Gant = calc_Gant_simple(g_a)
      T_Rx = calc_TRx(self.T_Rx,auxparams.outfreq)
      Tmod = [Gant * oneT + T_Rx for oneT in sky_temp]
      Tsys = Tmod
    elif self.inputtype == 3:
      Tmod = sky_temp
      if 'gamma_ant' in precomputed:
        g_a = precomputed['gamma_ant']
      else:
        g_a = self.gamma_ant.calc_resp(auxparams.outfreq)
      if self.noisepars.expanded == False:
        self.noisepars.expand(auxparams.outfreq)
      Z_ant = 0.5 * self.noisepars.Zzero * (1 + g_a) / (1 - g_a)
      T_amp0 = calc_Tamp(self.noisepars.Zzero,self.noisepars,auxparams.outfreq)
      T_amp1 = calc_Tamp(Z_ant,self.noisepars,auxparams.outfreq)
      if 'S' in precomputed:
        S = precomputed['S']
      else:
        S = calc_S(self.S,auxparams.outfreq)
      gamma_L = calc_gamma_L(self.gamma_L,auxparams.outfreq)
      G_amp0 = calc_Gamp0(S[:,1,0],S[:,1,1],gamma_L)
      g = calc_gainratio(S,g_a,gamma_L)
      G_amp1 = g * G_amp0
      T_Rx = calc_TRx(self.T_Rx,auxparams.outfreq)
      Tsys = calc_Tsys(Tmod,T_amp0,T_amp1,G_amp0,G_amp1,
                       self.noisepars.Tamb,T_Rx)
      if return_more:
        return Tmod,Tsys,T_amp0,T_amp1,G_amp0,G_amp1,self.noisepars.Tamb,T_Rx
    return Tmod,Tsys

def calc_TRx(coeffs,outfreq):
  """For now, this is just a trivial function that returns its input expanded
  into an array."""
  return coeffs * np.ones(outfreq.size)


def calc_gamma_L(coeffs,outfreq):
  """For now, this is just a trivial function that returns its input expanded
  into an array."""
  return coeffs * np.ones(outfreq.size)


def calc_Tamp(Z,noisepar_coeffs,outfreq):
  """Computes an amplifier temperature as a function of frequency for a given
  impedance."""
  if not noisepar_coeffs.expanded:
    noisepar_coeffs.expand(outfreq)
  
  return (noisepar_coeffs.Tmin + noisepar_coeffs.Tref * 
          np.square(abs(Z - (noisepar_coeffs.Ropt + 1j*noisepar_coeffs.Xopt)))
          / (noisepar_coeffs.Zzero * noisepar_coeffs.Rn))
  
def calc_gainratio(S,gamma_ant,gamma_L):
  """Computes g = Gamp1/Gamp0, assuming that gamma_ant and gamma_L have already
  been expanded into complex arrays of length nchannels, and that S is a 
  complex array of dimensions [nchannels,2,2], i.e. an S-matrix that has had
  its polynomial coefficients expanded out already."""
  gamma_in = (S[:,0,0] + S[:,0,1] * S[:,1,0] * gamma_L
              / (1 - S[:,1,1] * gamma_L))
  return ((1 - np.square(abs(gamma_ant))) / 
          np.square(abs(1 - gamma_ant * gamma_in)))
  

def calc_gamma_ant(gamma_ant_coeffs,outfreq):
  """Computes the reflection coefficient of the antenna. This routine only
  included for compatibility with others that use the procedural syntax."""
  return gamma_ant_coeffs.calc_resp(outfreq)
  

def calc_Gamp0(S21,S22,gamma_L):
  """Computes Gamp0, assuming that S21, S22 and gamma_L have already been
  expanded into arrays of length nchannels. Note that the matrix notation for
  the S-matrix assumes that matrix indices start at unity!"""
  return (np.square(abs(S21)) * (1 - np.square(abs(gamma_L))) / 
         np.square(abs(1 - S22*gamma_L)))


def calc_S(coeffs_mat,outfreq):
  """Computes the scattering matrix, S, a complex array of dimension
  [nchannels,2,2], from the polynomial coefficients off the real and 
  imaginary aprts of its entries."""
  S = np.empty([outfreq.size,2,2],np.complex)
  for i in [0,1]:
    for j in [0,1]:
      re_tmp = np.polyval(coeffs_mat[i][j].coeffs[:,0],outfreq)
      im_tmp = np.polyval(coeffs_mat[i][j].coeffs[:,1],outfreq)
      S[:,i,j] = re_tmp + 1j * im_tmp
  return S

def calc_Gant_simple(gamma_ant):
  return 1 - np.square(np.absolute(gamma_ant))

def calc_Tsys(T_ant,T_amp0,T_amp1,G_amp0,G_amp1,T_amb,T_Rx):
  T_other = T_amp1 + T_Rx/G_amp1 + T_amb + T_amp0 + T_Rx/G_amp0
  return [oneT + T_other for oneT in T_ant]
