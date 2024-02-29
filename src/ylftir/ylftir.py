import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import requests
from scipy.signal import argrelextrema, savgol_filter
from scipy.optimize import curve_fit

def import_data_url(url):
  try:
    response = requests.get(url)
    if url.endswith('.dpt'):
      x, y = np.transpose(np.genfromtxt(response.text.splitlines()))
    elif url.endswith('.csv'):
      x, y = np.transpose(np.genfromtxt(response.text.splitlines(), delimiter=','))

    # sort y w.r.t. x
    # NOTE: if files are guaranteed to be in order (or reverse order) with respect to wavenumber, this is unnecessary
    x_ind = np.argsort(x)
    x, y = x[x_ind], y[x_ind]
    return x, y
  except IOError:
    print(f'Unable to open file at {url}')
    return None, None
  except Exception as e:
    print(f'Error: {e}')
    return None, None

def import_data_file(file):
  try:
    f = open(file, 'r')
  except IOError:
    print(f'Unable to open file at {file}')
    return None, None
  except Exception as e:
    print(f'Error: {e}')
    return None, None
  else:
    with f:
      if file.endswith('.dpt'):
        x, y = np.transpose(np.genfromtxt(f.read().splitlines()))
      elif file.endswith('.csv'):
        x, y = np.transpose(np.genfromtxt(file, delimiter=','))
      else:
        print(f'Error: Unrecognized filetype')

      # sort y w.r.t. x
      # NOTE: if files are guaranteed to be in order (or reverse order) with respect to wavenumber, this is unnecessary
      x_ind = np.argsort(x)
      x, y = x[x_ind], y[x_ind]
      return x, y

def crop(x, y, range):
  diffs1 = np.abs(x - range[0])
  diffs2 = np.abs(x - range[1])
  i1 = np.argmin(diffs1)
  i2 = np.argmin(diffs2)
  x = x[i1:i2]
  y = y[i1:i2]
  return x, y

def gaussian(x, position, amplitude, width):
  return amplitude * np.exp(-0.5 * ((x - position) / width) ** 2)

def gaussian_area(position, amplitude, width):
  return amplitude * np.sqrt(2 * np.pi) * width

class Optim_Gaussian:
  def __init__(self, positions):
    self.positions = positions

  def __call__(self, x, *args):
    '''
    args should be in the form amplitude, amplitude, ..., width, width, ... etc.
    '''
    amplitudes, widths = np.array(args).reshape(2, -1)
    return sum([gaussian(x, self.positions[i], amplitudes[i], widths[i]) for i in range(len(self.positions))])

def lorentzian(x, position, amplitude, width):
  return (amplitude * width**2) / (np.power(x - position, 2) + width**2)

def lorentzian_area(position, amplitude, width):
  return amplitude * np.pi * width

class Optim_Lorentzian:
  def __init__(self, positions):
    self.positions = positions

  def __call__(self, x, *args):
    '''
    args should be in the form amplitude, amplitude, ..., width, width, ... etc.
    '''
    amplitudes, widths = np.array(args).reshape(2, -1)
    return sum([lorentzian(x, self.positions[i], amplitudes[i], widths[i]) for i in range(len(self.positions))])

def linear_baseline_correction(x, y):
  slope = (y[-1] - y[0]) / (x[-1] - x[0])
  baseline = y[0] + slope * (x - x[0])
  return y - baseline, baseline

def find_nodes(x, y, window_size=50):
  nodes = [0]
  for i in range(1, len(y)-1):
    wavenumber_range = np.abs(x - x[i])
    window_indices = np.where(wavenumber_range <= window_size / 2)
    if y[i] == np.min(y[window_indices]):
      nodes.append(i)
  nodes.append(len(y)-1)
  return np.array(nodes)

def nodes_to_indices(x, nodes):
  nodes = np.array(nodes)
  tx = np.repeat(x[:,None], len(nodes), axis=1)
  tn = np.repeat(nodes[None,:], len(x), axis=0)
  return np.argmin(np.abs(tx - tn), axis=0)

def nodal_baseline_correction(x, y, nodes=None, window_size=50):
  if nodes is None:
    nodes = find_nodes(x, y, window_size)
  else:
    nodes = np.sort(nodes_to_indices(x, nodes))

  baseline = np.zeros_like(y)
  for i in range(len(nodes)-1):
    i1 = nodes[i]
    i2 = nodes[i+1]
    x1, y1 = x[i1], y[i1]
    x2, y2 = x[i2], y[i2]

    slope = (y2 - y1) / (x2 - x1)
    baseline[i1:i2+1] = y1 + slope * (x[i1:i2+1] - x1)

  return y - baseline, baseline

def find_peaks_d1(x, y, smoothing_size=20):
  '''
  Estimate number and positions of peaks using smoothed first derivatives
  '''
  y = savgol_filter(y, smoothing_size, 3)
  #dy = savgol_filter(np.gradient(y, x), 20, 2)

  inds = argrelextrema(y, np.greater)[0]

  return len(inds), x[inds]

def find_peaks_d2(x, y, window_size=35, smoothing_size=20):
  '''
  Estimate number and positions of peaks using smoothed second derivatives

  Currently uses Savitzky-Golay polynomial smoothing
  '''
  y = savgol_filter(y, smoothing_size, 3)
  dy = savgol_filter(np.gradient(y, x), smoothing_size, 2)
  d2y = savgol_filter(np.gradient(dy, x), smoothing_size, 1)

  inds = find_nodes(x, d2y, window_size)[1:-1]

  return len(inds), x[inds]

def plot_d2(x, y, smoothing_size=20):
  y = savgol_filter(y, smoothing_size, 3)
  dy = savgol_filter(np.gradient(y, x), smoothing_size, 2)
  d2y = savgol_filter(np.gradient(dy, x), smoothing_size, 1)

  plt.figure(figsize=(8,6))
  plt.plot(x, d2y)

def plot_baseline(x, y, nodes=None, window_size=50):
  y_n, _ = nodal_baseline_correction(x, y, nodes=nodes, window_size=window_size)

  plt.figure(figsize=(8,6))
  plt.plot(x, y_n)
  plt.grid()

def plot_simulated(x, y, baseline_residual, simulated, gaussians):
  y_n = y - baseline_residual

  plt.figure(figsize=(8,6))
  plt.plot(x, y_n, 'k', label='Input data')
  for i in range(len(gaussians)):
    plt.plot(x, gaussians[i], 'b--')
  plt.plot(x, simulated, 'r--', label='Deconvoluted curve')
  plt.grid()
  plt.xlabel('Wavenumber (1/cm)', fontsize=16)
  plt.ylabel('Absorbance', fontsize=16)
  plt.tick_params(left=False, labelleft=False)
  plt.gca().invert_xaxis()
  plt.legend()

def plot_uncorrected(x, y, baseline_residual, simulated, gaussians):
  plt.figure(figsize=(8,6))
  plt.plot(x, y, color='k', label='Input data')
  for i in range(len(gaussians)):
    plt.plot(x, gaussians[i] + baseline_residual, 'b--')
  plt.plot(x, simulated + baseline_residual, 'r--', label='Deconvoluted curve')
  plt.grid()
  plt.legend()

def ftir_deconvolution(
    x, y,
    fit_func='gaussian', baseline='nodal', peak_finder='second_derivative',
    positions=None,
    nodes=None, window_size=50,
    peak_window_size=35, smoothing_size=20
    ):
  '''
  Computes full deconvolution of input data given deconvolution parameters
  `x` - wavenumbers, sorted
  `y` - absorbance
  `fit_func` - `'gaussian'` or `'lorentzian'`; the fitting function used. defaults to `'gaussian'`
  `baseline` - `'nodal'` or `'linear'`; baseline correction used. defaults to `'linear'`
  `peak_finder` - `'first_derivative'` or `'second_derivative'`; the method for finding peaks if `n` and `positions` is not provided
  `positions` - list of positions for each peak
  `nodes` - list of wavenumbers in x to consider as nodes for nodal baseline correction
  `window_size` - size of window when using nodal baseline correction
  `peak_window_size` - size of window for sencond derivative peak finding
  `smoothing_size` - number of points for Savitzky-Golay polynomial smoothing
  '''
  # baseline correction
  if baseline == 'nodal':
    y, residual = nodal_baseline_correction(x, y, nodes=nodes, window_size=window_size)
  elif baseline == 'linear':
    y, residual = linear_baseline_correction(x, y)

  # if positions not provided, then we need to do peak finding
  if positions is None:
    if peak_finder == 'second_derivative':
      n, positions = find_peaks_d2(x, y, window_size=peak_window_size, smoothing_size=smoothing_size)
    elif peak_finder == 'first_derivative':
      n, positions = find_peaks_d1(x, y, smoothing_size=smoothing_size)
    print(f'Using {n} peaks at {positions}')
  else:
    n = len(positions)

  # setup and solve optimization problems
  init = [*[1]*n, *[0.1]*n]

  scale_x = np.max(x) - np.min(x)
  scale_y = np.max(y) - np.min(y)

  if fit_func == 'gaussian':
    problem = Optim_Gaussian(positions / scale_x)
  elif fit_func == 'lorentzian':
    problem = Optim_Lorentzian(positions / scale_x)
  solution, cov = curve_fit(problem, x / scale_x, y, init, maxfev=100000, bounds=(0,np.inf))

  # return problem solution as numpy array of parameters
  amplitudes, widths = np.array(solution).reshape(2, -1)
  widths *= scale_x

  return np.array([positions, amplitudes, widths]), residual

def simulated_gaussians(x, gaussians):
  '''
  Returns the sum of input gaussians along input x
  `x` - wavenumbers, sorted
  `gaussians` - list in shape (3,N) where N is the number of gaussians, in the form [positions, amplitudes, widths]
  '''
  positions, amplitudes, widths = gaussians
  res = np.array([gaussian(x, positions[i], amplitudes[i], widths[i]) for i in range(len(positions))])
  return np.sum(res, axis=0), res

def simulated_lorentzians(x, lorentzians):
  '''
  Returns the sum of input lorentzians along input x
  `x` - wavenumbers, sorted
  `lorentzians` - list in shape (3,N) where N is the number of lorentzians, in the form [positions, amplitudes, widths]
  '''
  positions, amplitudes, widths = lorentzians
  res = np.array([lorentzian(x, positions[i], amplitudes[i], widths[i]) for i in range(len(positions))])
  return np.sum(res, axis=0), res
