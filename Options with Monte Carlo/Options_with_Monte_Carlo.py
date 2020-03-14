import scipy as sp
import scipy.stats as si
from numpy import zeros
import numpy as np

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

S0=40.
K=40.
H=42.
T=1.
r=0.05
q=0.
sigma=0.2
n_simulation=100 # should be number of terminal prices
n_steps=100. # why does this lead to the number of prices?
#sp.random.seed(1234)

### PRICING OPTIONS WITH BLACK SCHOLES ###
def comp_bs_call(S0,K,T,r,q,sigma):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    bs_call = (S0 * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    
    print("European long call option Black-Scholes:", np.round(bs_call, 2))

def comp_bs_put(S0,K,T,r,q,sigma):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    bs_put = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S0 * np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0))
    
    print("European long put option Black-Scholes:", np.round(bs_put, 2))

# test if BS calculation works
comp_bs_call(S0,K,T,r,q,sigma)
comp_bs_put(S0,K,T,r,q,sigma)

### PRICING OPTIONS USING MONTE CARLO SIMULATION ###

# Call
dt=T/n_steps # fraction of a year
call = np.zeros([n_simulation], dtype=float)
x = range(0, int(n_steps), 1)
for j in range(0, n_simulation):
    ST=S0
for i in x[:-1]:
    e=sp.random.normal()
    ST*=np.exp((r-0.5*sigma*sigma)*dt+sigma*e*np.sqrt(dt)) # simulation of ending prices
    call[j]=max(ST-K,0)
    call_price= np.mean(call)*np.exp(-r*T) # take average values of ending and calculate present value
    print('call price =', round(call_price, 3)) # returns number of simulations-1 (why?) call prices

# Put


### PRICING BARRIER OPTIONS USING MONTE CARLO SIMULAITON ###
def down_and_in_put(S0,K,H,T,r,sigma,n_simulation):
    n_steps=100.
    dt=T/n_steps
    total=0
    for j in range (0,n_simulation):
        ST=S0
        in_=False
        for i in range (0,int(n_steps)):
            e=sp.random.normal()
            ST*=np.exp((r-0.5*sigma*sigma)*dt+sigma*e*np.sqrt(dt))
            if ST<H:
                in_=True
                #print "ST=",ST
        # print "j=",j, "out=",out
        if in_==True:
            total+=bs_put(S0,K,T,r,q,sigma) # += add right operand to the left operan
    return total/n_simulation

print("Everything works")



