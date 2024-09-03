import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special

def coverage_prob(s_, method, num_samples=1000000):
    b_ = 3.2
    n_ = np.random.poisson(s_ + b_, num_samples)
    s_up_ = 0
    if method == "classical":
        s_up_ = 0.5 * special.chdtri(2 * (n_ + 1), 0.1) - b_
    elif method == "bayesian":
        s_up_ = 0.5 * special.chdtri(2 * (n_ + 1), 0.1 * (special.chdtrc(2 * (n_ + 1), 2 * b_))) - b_
    p_ = np.count_nonzero(s_up_ > s_) / num_samples
    return p_

p = 925000/1000000
p_classical = np.array([])
p_bayesian = np.array([])
s = np.linspace(0.1, 20, 200)
print('p:' )
print(p)

for i in s:
    p_classical = np.append(p_classical, coverage_prob(i, method="classical"))
    p_bayesian = np.append(p_bayesian, coverage_prob(i, method="bayesian"))

plt.plot(s, p_classical, color="blue", label="Classical")
plt.plot(s, p_bayesian, color="black", label="Bayesian")
plt.xlabel("s")
plt.ylabel("p")
plt.title("Coverage probability of 90% confidence interval for s")
plt.legend()
plt.show()