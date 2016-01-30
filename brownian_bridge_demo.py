try:
    import numpy as np
except ImportError:
    print("Requires numpy version 1.10.2 and above")

from hmc_tmg import HMCTruncGaussian


__author__ = 'Paul Tune'
__date__ = '30 Jan 2016'


def brownian_bridge_sampler(time, var, V0, VT, samples=1):
    # initialize covariance matrix
    cov_mtx = 2*np.identity(time-1)
    for i in range(time-2):
        cov_mtx[i,i+1] = -1
        cov_mtx[i+1,i] = -1

    r = np.zeros((time-1, 1))
    r[0,0] = V0/var
    r[time-2,0] = VT/var

    F = -np.identity(time-1)
    g = VT*np.ones((time-1,1))

    initial = (VT-2)*np.ones((time-1,1))

    hmc = HMCTruncGaussian()
    return hmc.generate_general_tmg(F, g, cov_mtx, r, initial, samples=samples, cov=False)

def main():
    # set simulation parameters
    time = 10
    var = 1
    V0 = -40
    VT = -20
    samples = 1

    # low noise regime
    sample_mtx_low = brownian_bridge_sampler(time, var, V0, VT, samples=samples)

    # high noise regime
    var = 5
    sample_mtx_high = brownian_bridge_sampler(time, var, V0, VT, samples=samples)

    # burn-in samples to discard
    burn_in_discard = 500

if __name__ == "__main__":
    main()



