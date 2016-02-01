try:
    import numpy as np
except ImportError:
    print("Requires numpy version 1.10.2 and above")

import numpy.linalg as lin
import sys

from random import gauss
from numpy.linalg import cholesky


__author__ = 'Paul Tune'
__date__ = "21 Jan 2016"

EPS = 1e-11     # this determines the precision of how close we want to be to the boundary


class HMCTruncGaussian(object):
    """
    Hamiltonian Markov Chain (HMC) generation of multivariate truncated normal random variables. Exact expressions
    are used in the generation. Based on Matlab code by Ari Pakman (https://github.com/aripakman/hmc-tmg).

    References:
    [1] Ari Pakman and Liam Paninski, "Exact Hamiltonian Monte Carlo for Truncated Multivariate Gaussians",
        http://arxiv.org/abs/1208.4118
    """
    def generate_simple_tmg(self, mean, std_dev, samples=1):
        """
        Generates samples of truncated Gaussian distributed random vectors with covariance matrix structure identity
        matrix times std_dev**2. Random vector length will be equal to the mean vector length, specified as a parameter.

        Example usage:

            >> mean = [0.1] * 5
            >> std_dev = 1
            >> print(HMCTruncGaussian().generate_simple_tmg(mean, std_dev))
            [[1.5393077420852723, 0.83193549862758009, 0.17057082476061466, 0.35605405861148831, 0.54828265215645966]]

        :param mean: mean vector of distribution (note: this is the mean after truncation of a normal distribution)
        :param std_dev: standard deviation of distribution
        :param samples: number of samples to output (default=1).
        :return: list of samples
        """

        # output will be generated as a matrix of size |mean| by |samples|
        # convert all to numpy matrix for easier handling
        dim = len(mean)     # dimension of mean vector; each sample must be of this dimension

        # define all vectors in column order; may change to list for output
        mu = np.matrix(mean).transpose()
        if any(mu) < 0:
            print("Error: mean vector must be positive")
            return

        # check standard deviation
        if std_dev <= 0:
            print("Error: standard deviation must be positive")
            return

        g = np.matrix(mean).transpose()
        initial_sample = np.divide(np.ones((dim, 1)) - mu, std_dev) # we choose a simple non-negative vector
        sample_matrix = []

        # more for debugging purposes
        if any(initial_sample + g) < 0:
            print("Error: inconsistent initial condition")
            return

        # count total number of times boundary has been touched
        bounce_count = 0

        # generate samples
        for i in range(samples):
            print("Normal")
            stop = False
            j = -1
            # use gauss because it's faster
            initial_velocity = np.matrix([gauss(0, 1) for _ in range(dim)]).transpose()
            # print(initial_velocity)
            # initial_velocity = np.matrix('1.4090; 1.4172; 0.6715; -1.2075')
            # initial_velocity = np.matrix('-0.46510217; -0.34660608; -1.17232004; -1.89907886')
            # initial_velocity = np.matrix('0.38491682; 1.27530709; 0.7218227; -0.00850574; 0.22724687')
            previous = initial_sample.__copy__()

            x = previous.__copy__()
            T = np.pi/2
            tt = 0

            while True:
                a = np.real(initial_velocity.__copy__())
                b = x.__copy__()

                u = np.sqrt(std_dev**2*np.square(a) + std_dev**2*np.square(b))
                # has to be arctan2 not arctan
                phi = np.arctan2(-std_dev**2*a, std_dev**2*b)

                # print(a)
                # find the locations where the constraints were hit
                pn = np.abs(np.divide(g, u))
                t1 = sys.maxsize*np.ones((dim, 1))

                collision = False
                inds = [-1] * dim
                for k in range(dim):
                    if pn[k] <= 1:
                        collision = True
                        pn[k] = 1
                        # compute the time the coordinates hit the constraint wall
                        t1[k] = -1*phi[k] + np.arccos(np.divide(-1*g[k], u[k]))
                        inds[k] = k
                    else:
                        pn[k] = 0

                if collision:
                    # if there was a previous reflection (j > -1)
                    # and there is a potential reflection at the sample plane
                    # make sure that a new reflection at j is not found because of numerical error
                    if j > -1:
                        if pn[j] == 1:
                            cum_sum_pn = np.cumsum(pn).tolist()
                            temp = cum_sum_pn[0]

                            index_j = int(temp[j])-1
                            tt1 = t1[index_j]

                            if np.abs(tt1) < EPS or np.abs(tt1 - 2*np.pi) < EPS:
                                t1[index_j] = sys.maxsize

                    mt = np.min(t1)

                    # update j
                    j = inds[int(np.argmin(t1))]
                else:
                    mt = T

                # update travel time
                tt += mt

                if tt >= T:
                    mt -= tt - T
                    stop = True

                # print(a)
                # update position and velocity
                x = np.multiply(a, np.sin(mt)) + np.multiply(b, np.cos(mt))
                v = np.multiply(a, np.cos(mt)) - np.multiply(b, np.sin(mt))

                if stop:
                    break

                # update new velocity
                initial_velocity[j] = -v[j]
                for k in range(dim):
                    if k != j:
                        initial_velocity[k] = v[k]

                # print(initial_velocity)
                bounce_count += 1

            sample = std_dev*x + mu
            sample = sample.transpose().tolist()
            sample_matrix.append(sample[0])

        return sample_matrix

    def generate_general_tmg(self, Fc, gc, M, mean_r, initial, samples=1, cov=True):
        """
        Generates samples of truncated Gaussian distributed random vectors with general covariance matrix under
        constraint

        Fc * x + g >= 0.

        Random vector length will be equal to the mean vector length, specified as a parameter.

        Example usage - generation of non-negative truncated normal random vectors of size 5, with identity
        covariance matrix:
            >> import numpy as np
            >> size = 5
            >> mean = [0.1] * size
            >> cov_mtx = np.identity(size)
            >> Fc = np.identity(size)
            >> g = np.zeros((size,1))
            >> initial = np.ones((size,1))
            >> print(HMCTruncGaussian().generate_general_tmg(Fc, g, cov_mtx, mean, initial))
            [[1.5393077420852723, 0.83193549862758009, 0.17057082476061466, 0.35605405861148831, 0.54828265215645966]]

        :param Fc: constraint matrix
        :param g: constraint vector
        :param mean: mean vector of distribution (note: this is the mean after truncation of a normal distribution)
        :param cov_mtx: covariance matrix of distribution
        :param initial: initial/starting point
        :param samples: number of samples to output (default=1).
        :return: list of samples
        """
        # sanity check
        s = gc.shape[0]
        if Fc.shape[0] != s:
            print("Error: constraint dimensions do not match")
            return

        try:
            R = cholesky(M)
        except lin.LinAlgError:
            print("Error: covariance or precision matrix is not positive definite")
            return

        # using covariance matrix
        if cov:
            mu = np.matrix(mean_r)
            if mu.shape[1] != 1:
                mu = mu.transpose()

            g = np.matrix(gc) + np.matrix(Fc)*mu
            F = np.matrix(Fc)*R.transpose()
            initial_sample = lin.solve(R.transpose(), initial - mu)
        # using precision matrix
        else:
            r = np.matrix(mean_r)
            if r.shape[1] != 1:
                r = r.transpose()

            mu = lin.solve(R, lin.solve(R.transpose(), r))
            g = np.matrix(gc) + np.matrix(Fc)*mu
            F = lin.solve(R, np.matrix(Fc))
            initial_sample = initial - mu
            initial_sample = R*initial_sample

        dim = len(mu)     # dimension of mean vector; each sample must be of this dimension

        # define all vectors in column order; may change to list for output
        sample_matrix = []

        # more for debugging purposes
        if any(F*initial_sample + g) < 0:
            print("Error: inconsistent initial condition")
            return

        # count total number of times boundary has been touched
        bounce_count = 0

        # squared Euclidean norm of constraint matrix columns
        Fsq = np.sum(np.square(F), axis=0)
        Ft = F.transpose()
        # generate samples
        for i in range(samples):
            print("General HMC")
            stop = False
            j = -1
            # use gauss because it's faster
            initial_velocity = np.matrix([gauss(0, 1) for _ in range(dim)]).transpose()
            # print(initial_velocity)
            # initial_velocity = np.matrix('1.4090; 1.4172; 0.6715; -1.2075')
            # initial_velocity = np.matrix('-0.46510217; -0.34660608; -1.17232004; -1.89907886')
            # initial_velocity = np.matrix('0.38491682; 1.27530709; 0.7218227; -0.00850574; 0.22724687')
            previous = initial_sample.__copy__()

            x = previous.__copy__()
            T = np.pi/2
            tt = 0

            while True:
                a = np.real(initial_velocity.__copy__())
                b = x.__copy__()

                fa = F*a
                fb = F*b

                u = np.sqrt(np.square(fa) + np.square(fb))
                # has to be arctan2 not arctan
                phi = np.arctan2(-fa, fb)

                # print(a)
                # find the locations where the constraints were hit
                pn = np.abs(np.divide(g, u))
                t1 = sys.maxsize*np.ones((dim, 1))

                collision = False
                inds = [-1] * dim
                for k in range(dim):
                    if pn[k] <= 1:
                        collision = True
                        pn[k] = 1
                        # compute the time the coordinates hit the constraint wall
                        t1[k] = -1*phi[k] + np.arccos(np.divide(-1*g[k], u[k]))
                        inds[k] = k
                    else:
                        pn[k] = 0

                if collision:
                    # if there was a previous reflection (j > -1)
                    # and there is a potential reflection at the sample plane
                    # make sure that a new reflection at j is not found because of numerical error
                    if j > -1:
                        if pn[j] == 1:
                            cum_sum_pn = np.cumsum(pn).tolist()
                            temp = cum_sum_pn[0]

                            index_j = int(temp[j])-1
                            tt1 = t1[index_j]

                            if np.abs(tt1) < EPS or np.abs(tt1 - 2*np.pi) < EPS:
                                t1[index_j] = sys.maxsize

                    mt = np.min(t1)

                    # update j
                    j = inds[int(np.argmin(t1))]
                else:
                    mt = T

                # update travel time
                tt += mt

                if tt >= T:
                    mt -= tt - T
                    stop = True

                # print(a)
                # update position and velocity
                x = a*np.sin(mt) + b*np.cos(mt)
                v = a*np.cos(mt) - b*np.sin(mt)

                if stop:
                    break

                # update new velocity
                reflected = F[j,:]*v/Fsq[0,j]
                initial_velocity = v - 2*reflected[0,0]*Ft[:,j]

                bounce_count += 1

            # need to transform back to unwhitened frame
            if cov:
                sample = R.transpose()*x + mu
            else:
                sample = lin.solve(R, x) + mu

            sample = sample.transpose().tolist()
            sample_matrix.append(sample[0])

        return sample_matrix



if __name__ == "__main__":

    tmg = HMCTruncGaussian()
    # mean = [0.1, 0.2, 0.35, 0.62]

    size = 5
    mean = [0.1] * size
    # print(mean)
    std_deviation = 1
    print(tmg.generate_simple_tmg(mean, std_deviation, samples=1))

    # covariance matrix
    cov_mtx = std_deviation**2 * np.identity(size)

    # constraints
    F = np.identity(size)
    g = np.zeros((size,1))
    print(tmg.generate_general_tmg(F, g, cov_mtx, mean, np.ones((size,1)), cov=False, samples=1))




