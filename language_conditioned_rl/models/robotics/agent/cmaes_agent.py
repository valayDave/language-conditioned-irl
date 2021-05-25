"""
This module defines the BasisModel base class, which defines general purpose functions used by all basis models.

@author Joseph Campbell <jacampb1@asu.edu>, Interactive Robotics Lab, Arizona State University
Thank You Joe!

This is brilliantly written code!. Small Refactor to comments based on the code. 
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial
import scipy.linalg
import scipy.optimize
import scipy.sparse
import sklearn.preprocessing

DTYPE = np.float64
DEFAULT_NUM_SAMPLES = 100


class BasisModel(object):
    def __init__(self, degree, observed_dof_names):
        """
        The BasisModel class is a base level class defining general methods that are used by all implemented basis models.
        This class corresponds to a shared basis space for 1 or more degrees of freedom.
        That is, every DoF modeled by this basis space uses the same set of basis functions with the same set of parameters.
        
        Initialization method for the BasisModel class.
        @param degree The degree of this basis model. Corresponds to the number of basis functions distributed throughout the domain.
        @param observed_dof_names The names for the observed degrees of freedom that use this basis space. The length is used to compute the number of observed degrees of freedom.
        """
        self.observed_dof_names = observed_dof_names
        self._degree = degree
        self.num_observed_dof = len(observed_dof_names)
        self._num_blocks = self.num_observed_dof
        self.block_prototype = scipy.linalg.block_diag(
            *np.tile(np.ones((degree, 1)), (1, self.num_observed_dof)).T).T

    def get_block_diagonal_basis_matrix(self, x, out_array=None, start_row=0, start_col=0):
        """
        Gets the block diagonal basis matrix for the given phase value(s).
        Used to transform vectors from the basis space to the measurement space.
        @param x Scalar of vector of dimension T containing the phase values to use in the creation of the block diagonal matrix.
        @param out_array Matrix of dimension greater to or equal than (degree * num_observed_dof * T) x num_observed_dof in which the results are stored. If none, an internal matrix is used.
        @param start_row A row offset to apply to results in the block diagonal matrix.
        @param start_col A column offset to apply to results in the block diagonal matrix.
        @return block_matrix Matrix of dimension greater to or equal than (degree * num_observed_dof * T) x num_observed_dof containing the block diagonal matrix.
        """
        if(out_array is None):
            out_array = self.block_prototype

        basis_funcs = self.get_basis_functions(x)
        for block_index in range(self._num_blocks):
            out_array[start_row + block_index * self._degree: start_row +
                      (block_index + 1) * self._degree, start_col + block_index: start_col + block_index + 1] = basis_funcs

        return out_array

    def get_block_diagonal_basis_matrix_derivative(self, x, out_array=None, start_row=0, start_col=0):
        """
        Gets the block diagonal basis derivative matrix for the given phase value(s).
        Used to transform vectors from the derivative basis space to the measurement space.
        @param x Scalar containing the phase value to use in the creation of the block diagonal matrix.
        @param out_array Matrix of dimension greater to or equal than (degree * num_observed_dof) x num_observed_dof in which the results are stored. If none, an internal matrix is used.
        @param start_row A row offset to apply to results in the block diagonal matrix.
        @param start_col A column offset to apply to results in the block diagonal matrix.
        @return block_matrix Matrix of dimension greater to or equal than (degree * num_observed_dof) x num_observed_dof containing the block diagonal matrix.
        """
        if(out_array is None):
            out_array = self.block_prototype

        basis_funcs = self.get_basis_function_derivatives(x)
        for block_index in range(self._num_blocks):
            out_array[start_row + block_index * self._degree: start_row +
                      (block_index + 1) * self._degree, start_col + block_index: start_col + block_index + 1] = basis_funcs

        return out_array

    def get_weighted_vector_derivative(self, x, weights, out_array=None, start_row=0, start_col=0):
        """
        Gets the weighted vector derivatives corresponding to this basis model for the given basis state.
        @param x Scalar containing the phase value to use in the creation of the block diagonal matrix.
        @param weights Vector of dimension degree * num_observed_dof containing the weights for this basis model.
        @param out_array Matrix of dimension greater to or equal than 1 x degree in which the results are stored. If none, an internal matrix is used.
        @param start_row A row offset to apply to results in the block diagonal matrix.
        @param start_col A column offset to apply to results in the block diagonal matrix.
        @return block_matrix Matrix of dimension greater to or equal than (degree * num_observed_dof) x num_observed_dof containing the block diagonal matrix.
        """
        if(out_array is None):
            out_array = np.zeros((1, self._degree))

        out_row = start_row
        basis_func_derivs = self.get_basis_function_derivatives(x[0])

        # temp_weights = self.inverse_transform(weights)
        temp_weights = weights

        for degree in range(self._num_blocks):
            offset = degree * self._degree
            out_array[out_row, start_col + degree] = np.dot(
                basis_func_derivs.T, temp_weights[offset: offset + self._degree])

        return out_array

    def fit_basis_functions_linear_closed_form(self, x, y):
        """
        Fits the given trajectory to this basis model via least squares.
        @param x Vector of dimension T containing the phase values of the trajectory.
        @param y Matrix of dimension num_observed_dof x T containing the observations of the trajectory.
        @param coefficients Vector of dimension degree * num_observed_dof containing the fitted basis weights.
        """
        basis_matrix = self.get_basis_functions(x).T

        reg_lambda = 0.001

        # The following two methods are equivalent, but the scipy version is more robust. Both are calculating the OLS solution to Ax = B.
        coefficients = np.linalg.solve(np.dot(basis_matrix.T, basis_matrix) + reg_lambda * np.identity(
            basis_matrix.shape[1]), np.dot(basis_matrix.T, y)).T  # .flatten()
        # coefficients = scipy.linalg.lstsq(basis_matrix, y)[0].T

        return coefficients

    def apply_coefficients(self, x, coefficients, deriv=False):
        """
        Applies the given weights to this basis model. Projects a basis state to the measurement space.
        @param x Scalar of vector of dimension T containing the phase values to project at.
        @param coefficients Vector of dimension degree * num_observed_dof containing the basis weights.
        @param deriv True to use basis function derivative, False to use regular basis functions.
        @return Vector of dimension num_observed_dof or matrix of dimension num_observed_dof x T if multiple phase values are given.
        """
        if(deriv):
            basis_funcs = self.get_basis_function_derivatives(x)
        else:
            basis_funcs = self.get_basis_functions(x)

        coefficients = coefficients.reshape((self._num_blocks, self._degree)).T

        result = np.dot(basis_funcs.T, coefficients)

        return result

    def plot(self):
        """Plots the unweighted linear basis model.
        """
        test_domain = np.linspace(0, 1, 100, dtype=DTYPE)
        test_range = self.get_basis_functions(test_domain)

        fig = plt.figure()

        for basis_func in test_range:
            plt.plot(test_domain, basis_func)

        fig.suptitle('Basis Functions')

        plt.show()

    def plot_derivative(self):
        """Plots the unweighted linear basis model.
        """
        test_domain = np.linspace(0, 1, 100, dtype=DTYPE)
        test_range = self.get_basis_function_derivatives(test_domain)

        fig = plt.figure()

        for basis_func in test_range:
            plt.plot(test_domain, basis_func)

        fig.suptitle('Basis Function Derivatives')

        plt.show(block=True)

    def plot_weighted(self, coefficients, coefficient_names):
        """Plots the weighted linear basis model.
        """
        test_domain = np.linspace(0, 1, 100, dtype=DTYPE)
        test_range = self.get_basis_functions(test_domain)

        for coefficients_dimension, name in zip(coefficients, coefficient_names):
            fig = plt.figure()

            for basis_func, coefficient in zip(test_range, coefficients_dimension):
                plt.plot(test_domain, basis_func * coefficient)

            fig.suptitle('Basis Functions For Dimension ' + name)

        plt.show(block=True)

    def observed_to_state_indices(self, observed_indices):
        state_indices = []

        try:
            for observed_index in observed_indices:
                state_indices.extend(range(
                    int(observed_index) * self._degree, (int(observed_index) + 1) * self._degree))
        except TypeError:
            state_indices.extend(range(
                int(observed_indices) * self._degree, (int(observed_indices) + 1) * self._degree))

        return np.array(state_indices, dtype=int)

    def observed_indices_related(self, observed_indices):
        return True

    def get_basis_functions(self, x, degree = None):
        raise NotImplementedError()

    def get_basis_function_derivatives(self, x, degree = None):
        raise NotImplementedError()

class GaussianModel(BasisModel):
    """
    The GaussianModel class implements a basis model consisting of Gaussian radial basis functions.
    """
    def __init__(self, degree, scale, observed_dof_names, start_phase = 0.0, end_phase = 1.01):
        """
        Initializer method for GaussianModel
        @param degree int The number of Gaussian basis functions which will be uniformly distributed throughout the space.
        @param scale float The variance of each Gaussian function. Controls the width.
        @param observed_dof_names array-like, shape (num_observed_dof, ). The names of all observed degrees of freedom.
        @param start_phase float The starting value from which the basis functions are uniformly distributed. The centers are calculated with linspace(start_phase, end_phase, degree).
        @param end_phase float The ending value from which the basis functions are uniformly distributed. The centers are calculated with linspace(start_phase, end_phase, degree).
        """
        super(GaussianModel, self).__init__(degree, observed_dof_names)

        # The variance of each Gaussian function. Controls the width.
        self.scale = scale
        # array-like, shape (degree, ) The center of each basis function uniformly distributed over the range [start_phase, end_phase]
        self.centers = np.linspace(start_phase, end_phase, self._degree, dtype = DTYPE)

        # dict A hash map storing previously computed basis values. The key is the requested phase value rounded to rounding_precision, and the value is the corresponding computed basis function values for that phase.
        # The first time a phase value is received, the value is computed and stored in the map. On subsequent calls, the stored value is returned (if the phase is within 4 digits), skipping the computation.
        # This map greatly aids in computation times for basis values, as often computations are requested thousands of times for very similar phase values.
        self.computed_basis_values = {}

        # float The precision with which phase values will be stored in the computed_basis_values hash map. A smaller value indicates a larger granularity, which means fewer computations will be performed at the expense of inaccurate basis function computations.
        # The default value is 10.0 ** 4, which indicates that the phase values are rounded to 4 digits after the decimal.
        self.rounding_precision = 10.0 ** 4

    def compute_basis_values(self, x):
        """
        Computes the basis values for the given phase value. The basis values are simply the evaluation of a Gaussian radial basis function at each center.

        @param x float The phase value for which to compute the basis values.

        @return array-like, shape(degree, ) The evaluated Gaussian radial basis functions for the given phase value.
        """
        # For computational efficiency...
        return np.exp(-((x - self.centers) ** 2) / (2.0 * self.scale))

    def get_basis_values(self, x):
        """
        Gets the basis function evaluations for the given phase value. If a phase value has previously been computed, returns the stored version. Otherwise computes it (via compute_basis_values) and stores it.
        @param x float The phase value for which to evaluate the basis functions.
        @return array-like, shape(degree, ) The evaluated Gaussian radial basis functions for the given phase value.
        """
        # Simple, optimized rounding function for pure speed.
        key = int(x * self.rounding_precision) / self.rounding_precision

        if(key in self.computed_basis_values):
            return self.computed_basis_values[key]
        else:
            values = self.compute_basis_values(x)
            self.computed_basis_values[key] = values
            return values

    def get_basis_functions(self, x, degree = None):
        """
        Gets the basis function evaluations for the given phase value(s). Essentially a vectorized wrapper to call get_basis_values for each phase value given.
        @param x array-like, shape(num_phase_values, ) or float. If array, a list of phase values for which to compute basis values. Otherwise, a single scalar phase value.
        @param degree int. Degree of this basis model.
        @return array-like, shape(degree, num_phase_values) or array-like, shape(degree, ) if x is a scalar. The evaluated Gaussian radial basis functions for the given phase value.
        """
        if(type(x) is np.ndarray):
            values = np.zeros((self.centers.shape[0], x.shape[0]))

            for value_idx in range(x.shape[0]):
                values[:, value_idx] = self.get_basis_values(x[value_idx])

            return values
        else:
            return self.get_basis_values(x)

    def get_basis_function_derivatives(self, x, degree = None):
        """
        Gets the evaluations for the derivative of the basis functions for the given phase value(s).
        This is necessary for the computation of the Jacobian matrix in the EKF filter.
        Unlike get_basis_functions, this function does not (currently) implement a hash map internally and so re-computes values everytime.
        Since the basis decompositions are simple linear combinations, the partial derivative of the combination with respect to each weight is simply the partial derivative of a single basis function due to the sum rule.
        This is the first order partial derivative with respect to x!
        It is used to compute the Jacobian matrix for filtering linear dynamical systems.
        Verified using wolfram alpha: d/dx a*e^(-(x-c)^2/(2*s)) + b*e^(-(x-d)^2/(2*s))
        @param x array-like, shape(num_phase_values, ) or float. If array, a list of phase values for which to compute basis derivative values. Otherwise, a single scalar phase value.
        @param degree int. Degree of this basis model.
        @return values array-like, shape(degree, num_phase_values) or array-like, shape(degree, ) if x is a scalar. The evaluated Gaussian radial basis function derivatives for the given phase value.
        """
        f = lambda x, c: (np.exp(-(np.array([x - y for y in c], dtype = DTYPE) ** 2) / (2.0 * self.scale)) * np.array([x - y for y in c], dtype = DTYPE)) / -self.scale

        return f(x, self.centers)