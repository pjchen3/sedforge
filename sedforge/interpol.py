import numpy as np
from itertools import product

def create_pixeltypegrid(grid_pars,grid_data):
   """
   Creates pixelgrid and arrays of axis values.
   
   Starting from:
      * grid_pars: 2D numpy array, 1 column per parameter, unlimited number of cols
      * grid_data: 2D numpy array, 1 column per variable, data corresponding to the rows in grid_pars
   
   example: interpolation in a 3D grid containing stellar evolution models. Say we have as
   input parameters mass, age and metalicity, and want to obtain teff and logg as variables.
   
   grid_pars =
      +------+-----+------+
      | mass | age | Fe/H |
      +------+-----+------+
      | 1.0  | 1.0 | -0.5 |
      +------+-----+------+
      | 2.0  | 1.0 | -0.5 |
      +------+-----+------+
      | 1.0  | 2.0 | -0.5 |
      +------+-----+------+
      | 2.0  | 2.0 | -0.5 |
      +------+-----+------+
      | 1.0  | 1.0 |  0.0 |
      +------+-----+------+
      | 2.0  | 1.0 |  0.0 |
      +------+-----+------+
      |...   |...  |...   |
      +------+-----+------+
      
   grid_data = 
      +------+------+
      | teff | logg |
      +------+------+
      | 5000 | 4.45 |
      +------+------+
      | 6000 | 4.48 |
      +------+------+
      |...   |...   |
      +------+------+
   
   The resulting grid will be rectangular and complete. This means that every
   combination of unique values in grid_pars should exist. If this is not the
   case, a +inf value will be inserted in grid_data at all locations that are 
   missing!

   
   :param grid_pars: Npar x Ngrid array of parameters
   :type grid_pars: array
   :param grid_data: Ndata x Ngrid array of data
   :type grid_data: array
   
   :return: axis values and pixelgrid
   :rtype: array, array
   """

   uniques = [np.unique(column, return_inverse=True) for column in grid_pars]
   #[0] are the unique values, [1] the indices for these to recreate the original array

   axis_values = [uniques_[0] for uniques_ in uniques]
   
   data_dim = np.shape(grid_data)[0]

   par_dims   = [len(uv[0]) for uv in uniques]

   par_dims.append(data_dim)
   pixelgrid = np.ones(par_dims)
   
   # We put np.inf as default value. If we get an inf, that means we tried to access
   # a region of the pixelgrid that is not populated by the data table
   pixelgrid[pixelgrid==1] = np.inf
   
   # now populate the multiDgrid
   indices = [uv[1] for uv in uniques]
   pixelgrid[tuple(indices)] = grid_data.T
   
   return axis_values, pixelgrid

def interpolate(p, axis_values, pixelgrid):
   """
   Interpolates in a grid prepared by create_pixeltypegrid().
   
   p is an array of parameter arrays
   each collumn contains the value for the corresponding parameter in grid_pars
   each row contains a set of model parameters for wich the interpolated values
   in grid_data are requested.
   
   example: continue with stellar evolution models used in create_pixeltypegrid
   
   p = 
      +------+-----+-------+
      | mass | age | Fe/H  | 
      +------+-----+-------+
      | 1.21 | 1.3 | 0.24  |
      +------+-----+-------+
      | 1.57 | 2.4 | -0.15 |
      +------+-----+-------+
      |...   |...  |...    |
      +------+-----+-------+
      
   >>> p = np.array([[1.21, 1.3, 0.24], [1.57, 2.4, -0.15]])
   >>> interpolate(p, axis_values, pixelgrid)
   >>> some output
   
   :param p: Npar x Ninterpolate array containing the points which to
             interpolate in axis_values
   :type p: array
   :param axis_values: output from create_pixeltypegrid
   :type axis_values: array
   :param pixelgrid: output from create_pixeltypegrid
   :type pixelgrid: array
   
   :return: Ndata x Ninterpolate array containing the interpolated values
            in pixelgrid
   :rtype: array
   
   """
   #-- The type of p is changes to the same type as in axis_values to catch possible rounding errors
   #   when comparing float64 to float32.
   for i, ax in enumerate(axis_values):
      p[i] = np.array(p[i], dtype = ax.dtype)
   
   #-- Build lower/upper node indices and multilinear weights.  Missing nodes
   #   are represented by non-finite values in pixelgrid.  Their weights are
   #   omitted and the remaining corner weights are renormalized, matching the
   #   sparse HDF5-grid interpolation used elsewhere in SedForge.
   bounds = []
   inside = np.ones(np.shape(p)[1], dtype=bool)
   for av_, val in zip(axis_values, p):
      if len(av_) == 1:
         tolerance = max(1e-10, 1e-8 * max(1.0, abs(float(av_[0]))))
         inside &= np.abs(val - av_[0]) <= tolerance
         bounds.append(((np.zeros_like(val, dtype=int),
                         np.ones_like(val, dtype=float)),))
         continue

      indices = np.searchsorted(av_, val)
      indices[indices == 0] = 1
      indices[indices == len(av_)] = len(av_) - 1
      lower = av_[indices - 1]
      step = av_[indices] - lower
      fraction = (val - lower) / step
      tolerance = max(
         1e-10,
         1e-8 * max(1.0, abs(float(av_[0])), abs(float(av_[-1]))),
      )
      inside &= (val >= av_[0] - tolerance) & (val <= av_[-1] + tolerance)
      bounds.append((
         (indices - 1, 1.0 - fraction),
         (indices, fraction),
      ))

   npoint = np.shape(p)[1]
   ndata = np.shape(pixelgrid)[-1]
   values = np.zeros((npoint, ndata), dtype=float)
   total_weight = np.zeros((npoint, ndata), dtype=float)

   for corner in product(*bounds):
      indices = tuple(item[0] for item in corner)
      weight = np.prod([item[1] for item in corner], axis=0)
      positive = inside & (weight > 0)
      if not np.any(positive):
         continue

      corner_values = np.asarray(pixelgrid[indices], dtype=float)
      finite = np.isfinite(corner_values) & positive[:, None]
      weighted = weight[:, None]
      contribution = np.zeros_like(corner_values, dtype=float)
      np.multiply(corner_values, weighted, out=contribution, where=finite)
      values += contribution
      total_weight += np.where(finite, weighted, 0.0)

   result = np.full((npoint, ndata), np.nan, dtype=float)
   valid = total_weight > 0
   result[valid] = values[valid] / total_weight[valid]
   return result.T
