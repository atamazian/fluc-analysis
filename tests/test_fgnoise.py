import numpy as np
from fluc_analysis.mfdfa import mfdfa, get_hurst
from fluc_analysis.fgnoise import fgnoise

def main():
    n = 1_000_000
    h = 0.7
    print(f'Generating a fractal gaussian noise sample with length of {n} and Hurst exponent of {h}...')
    x = fgnoise(n, h)
    scale = np.logspace(np.log10(10**1), np.log10(10**5), 30)
    print('Calculating Hurst exponent for the sample with DFA2...')
    _, F = mfdfa(x, scale, m=2)
    h_e = np.round(get_hurst(scale, F), 4)
    print(f'Hurst exponent for the generated noise was set to {h}')
    print(f'Calculated Hurst exponent for the sample is {h_e}')

if __name__ == '__main__':
    main()
