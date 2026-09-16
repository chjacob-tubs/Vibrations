import numpy as np
import pytest
import VibTools as vt
import os

tools_install_path = os.path.dirname(vt.__file__)
data_path = os.path.join(tools_install_path,"tests/test_data")


## H2O-Test-Setup
@pytest.fixture
def H2O_SNFsetup():
    path = os.path.join(data_path,'H2O/')
    resPySNF = vt.SNFResults(outname = os.path.join(path,'snf.out'),
                             restartname = os.path.join(path,'restart'),
                             coordfile = os.path.join(path,'coord'))
    resPySNF.read()
    a = range(0,len(resPySNF.modes.freqs))
    modes = resPySNF.modes.get_subset(a)
    return resPySNF, modes


## H2O-Spectrum-Test-Setup
@pytest.fixture
def H2O_Spectrum_setup(H2O_SNFsetup):
    res, modes = H2O_SNFsetup
    a = range(0,len(res.modes.freqs))
    modes = res.modes.get_subset(a)
    ints = res.get_ir_intensity(modes)
    vs = vt.VibSpectrum(res.modes.freqs,ints)
    return vs


def test_get_line_spectrum(H2O_Spectrum_setup) :
    # Created according to Bergmanns-Test scheme
    # Arrange
    vs = H2O_Spectrum_setup
    refs = [ 0., 0., 59.02232365, 0., 0., 
             2.67880051, 0., 0., 32.56993195, 0., 0.]
    x, y = vs.get_line_spectrum(0, 10000)
    for a,b in zip(refs,y):
        np.testing.assert_almost_equal(a, b, decimal=5)

def test_get_lorentz_spectrum(H2O_Spectrum_setup) :
    # Created according to Bergmanns-Test scheme
    # Arrange
    vs = H2O_Spectrum_setup
    x, y = vs.get_lorentz_spectrum(0, 10000)
    np.testing.assert_almost_equal(100000, len(x))
    np.testing.assert_almost_equal(100000, len(y))


def test_get_gaussian_spectrum(H2O_Spectrum_setup) :
    # Created according to Bergmanns-Test scheme
    # Arrange
    vs = H2O_Spectrum_setup
    x, y = vs.get_gaussian_spectrum(0, 10000)
    np.testing.assert_equal(100000, len(x))
    np.testing.assert_equal(100000, len(y))
    np.testing.assert_almost_equal(177.12348, y.sum(), decimal=3)

def test_get_rect_spectrum(H2O_Spectrum_setup) :
    # Created according to Bergmanns-Test scheme
    # Arrange
    vs = H2O_Spectrum_setup
    x, y = vs.get_rect_spectrum(0,10000,[[0,1],[1,2]])
    np.testing.assert_equal(100000, len(x))
    np.testing.assert_equal(100000, len(y))
    np.testing.assert_almost_equal(1341072.71, y.sum(), decimal=0)

def test_scale_range(H2O_Spectrum_setup) :
    # Created according to Bergmanns-Test scheme
    # Arrange
    vs = H2O_Spectrum_setup
    y = np.asarray([ 5., 0., 10, 5, 0., 10, 0., 5, 10, 5, 0.])
    x = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    a = y.sum()
    vs.scale_range(x, y, 1, 6, 2)
    b = y.sum()
    np.testing.assert_equal(a*1.5, b)

def test_get_band_maxima(H2O_Spectrum_setup) :
    # Created according to Bergmanns-Test scheme
    # Arrange
    vs = H2O_Spectrum_setup
    y = np.asarray([ 5., 0., 10, 5, 0., 10, 0., 5, 10, 5, 0.])
    x = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    for a,b in zip(vs.get_band_maxima(x,y), [2,5,8]):
        np.testing.assert_equal(a[0], b)

def test_get_band_minima(H2O_Spectrum_setup) :
    # Created according to Bergmanns-Test scheme
    # Arrange
    vs = H2O_Spectrum_setup
    y = np.asarray([ 5., 0., 10, 5, 0., 10, 0., 5, 10, 5, 0.])
    x = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    for a,b in zip(vs.get_band_minima(x,y), [1,4,6]):
        np.testing.assert_equal(a[0], b)


def test_get_plot(H2O_Spectrum_setup,H2O_SNFsetup):
    # Created according to Bergmanns-Test scheme
    # Arrange
    res, modes = H2O_SNFsetup
    vs = H2O_Spectrum_setup
    reffreqs = res.modes.freqs
    pl = vs.get_plot(0,10000)
    peak = pl.peaks
    freq = [peak[0][0][0],peak[1][0][0],peak[2][0][0]]
    for a,b in zip(freq, reffreqs):
        np.testing.assert_almost_equal(a, b, decimal=0)


def test_get_rect_plot(H2O_Spectrum_setup) :
    # Created according to Bergmanns-Test scheme
    # Arrange
    vs = H2O_Spectrum_setup
    refspec = vs.get_rect_spectrum(0,10000,bands = [[0,1],[1,2]])[1]
    pl = vs.get_rect_plot(0,10000, bands = [[0,1],[1,2]], bandnames = ['a','b'])
    spec = pl.spectra[1][1]
    for a,b in zip(spec, refspec):
        np.testing.assert_almost_equal(a,b)
