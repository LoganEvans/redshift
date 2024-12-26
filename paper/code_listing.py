import snpy
import numpy

# Use an arbitrary epoch for the spectral energy density template.
wave, flux = snpy.kcorr.get_SED(day=9)

# Use an arbitrary redshift.
z = 0.5

# Use the B and R filters from the Lick telescope as examples.
rest_filter = snpy.fset["Bkait"]
obs_filter = snpy.fset["Rkait"]

# Compute the desired rest magnitude.
m_b = rest_filter.synth_mag(wave, flux)

# Compute the observed magnitude after redshift effects.
m_r = obs_filter.synth_mag(wave, flux, z=z)

# TODO(lpe): Wait, why is m_r brighter than m_b? Shouldn't the fix term have
# the opposite sign?
# Huh... m_r is brighter than m_b even if SED is forced to be flat.

got_k = snpy.kcorr.K(wave, flux, rest_filter, obs_filter, z=z)[0]

# From the definition of k-corrections:
want_k = m_r - m_b

def print_results():
  """
  >>> print_results()
  got_k (-0.679138363247608) != want_k (-1.1193665108868114)
  2.5*log10(1+z) == 0.4402281476392031
  got_k - 2.5*log10(1+z) (-1.1193665108868112) == want_k (-1.1193665108868114)
  """
  print(f"got_k ({got_k}) != want_k ({want_k})")
  print(f"2.5*log10(1+z) == {2.5 * numpy.log10(1+z)}")
  fixed = got_k - 2.5*numpy.log10(1+z)
  print(f"got_k - 2.5*log10(1+z) ({fixed}) == want_k ({want_k})")
