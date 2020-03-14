import scipy as sp
import scipy.stats as si
from numpy import zeros
import numpy as np
import matplotlib.pyplot as plt

# Input parameters for Black-Scholes Formula
    # S0: spot price
    # K: strike price
    # H: Barrier
    # T: time to maturity
    # r: interest rate
    # q: rate of continuous dividend paying asset
    # sigma: volatility of underlying asset    
    # Nom: Nominal Value
    # CDS: Credit-Default Swap
    # c: Coupon
    
    # n_simulation: number of simulations
    # sp.random.seed: fix random numbers

stock_price_today = 9.15  # stock price at time zero
T =1.                     # maturity date (in years)
n_steps=100              # number of steps
mu =0.15                  # expected annual return 
sigma = 0.2               # volatility (annualized)
# sp.random.seed(12345)     # seed() 
n_simulation = 5          # number of simulations

dt=T/n_steps
S=np.zeros([n_steps])
x = range(0, int(n_steps), 1) # range(start, stop, step)
for j in range(0, n_simulation):
    S[0]= stock_price_today
for i in x[:-1]:
    e=np.random.normal()
    S[i+1]=S[i]+S[i]*(mu-0.5*sigma*sigma)*dt+sigma*S[i]*np.sqrt(dt)*e;
    plt.plot(x, S)
plt.text(0.2,0.8,'S0='+str(stock_price_today)+',mu='+str(mu)+',sigma='+str(sigma))
plt.text(0.2,0.76,'T='+str(T)+', steps='+str(int(n_steps)))
plt.title('Stock price (number of simulations = %d ' % n_simulation +')')
plt.xlabel('Total number of steps ='+str(int(n_steps)))
plt.ylabel('stock price')
plt.show()