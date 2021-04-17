import numpy as np
from fluc_analysis.mfdfa import mfdfa, get_hurst

def main():
    n = 1_000_000
    print(f'Generating a white noise sample with length of {n}...')
    x = np.random.normal(size=1_000_000)
    scale = np.logspace(np.log10(10**1), np.log10(10**5), 30)
    print('Calculating Hurst exponent for the sample with DFA2...')
    _, F = mfdfa(x, scale, m=2)
    hurst = np.round(get_hurst(scale, F), 4)
    print(f'Hurst exponent for the white noise is equal to 0.5')
    print(f'Calculated Hurst exponent for the sample is {hurst}')

if __name__ == '__main__':
    main()
    
