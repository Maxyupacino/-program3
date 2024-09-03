import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# Define the likelihood functions
def lnL0(m, data):
    ns = 48
    sigma = 1.4
    nb = 1520

    y = 0
    for i in data:
        c1 = (ns/(ns+nb)) * (1./np.sqrt(2*np.pi)) * (1./sigma) * np.exp(-((i-m)**2)/(2.*sigma**2))
        c2 = (1-(ns/(ns+nb))) * (i**-4.5 * (1/(2.492e-8)))
        y += np.log(c1 + c2)
    return y

def lnL1(m, data):
    ns = 462
    sigma = 1.5
    nb = 66180

    y = 0
    for i in data:
        c1 = (ns/(ns+nb)) * (1./np.sqrt(2*np.pi)) * (1./sigma) * np.exp(-((i-m)**2)/(2.*sigma**2))
        c2 = (1-(ns/(ns+nb))) * (i**-4.5 * (1/(2.492e-8)))
        y += np.log(c1 + c2)
    return y

def lnL2(m, data):
    ns = 704
    sigma = 2
    nb = 201599

    y = 0
    for i in data:
        c1 = (ns/(ns+nb)) * (1./np.sqrt(2*np.pi)) * (1./sigma) * np.exp(-((i-m)**2)/(2.*sigma**2))
        c2 = (1-(ns/(ns+nb))) * (i**-4.5 * (1/(2.492e-8)))
        y += np.log(c1 + c2)
    return y

def lnL3(m, data):
    ns = 594
    sigma = 2.7
    nb = 257082

    y = 0
    for i in data:
        c1 = (ns/(ns+nb)) * (1./np.sqrt(2*np.pi)) * (1./sigma) * np.exp(-((i-m)**2)/(2.*sigma**2))
        c2 = (1-(ns/(ns+nb))) * (i**-4.5 * (1/(2.492e-8)))
        y += np.log(c1 + c2)
    return y

def lnL4(m, data):
    ns = 11
    sigma = 1.6
    nb = 123

    y = 0
    for i in data:
        c1 = (ns/(ns+nb)) * (1./np.sqrt(2*np.pi)) * (1./sigma) * np.exp(-((i-m)**2)/(2.*sigma**2))
        c2 = (1-(ns/(ns+nb))) * (i**-4.5 * (1/(2.492e-8)))
        y += np.log(c1 + c2)
    return y

def lnL5(m, data):
    ns = 11
    sigma = 1.6
    nb = 382

    y = 0
    for i in data:
        c1 = (ns/(ns+nb)) * (1./np.sqrt(2*np.pi)) * (1./sigma) * np.exp(-((i-m)**2)/(2.*sigma**2))
        c2 = (1-(ns/(ns+nb))) * (i**-4.5 * (1/(2.492e-8)))
        y += np.log(c1 + c2)
    return y

def lnL6(m, data):
    ns = 35
    sigma = 1.7
    nb = 2238

    y = 0
    for i in data:
        c1 = (ns/(ns+nb)) * (1./np.sqrt(2*np.pi)) * (1./sigma) * np.exp(-((i-m)**2)/(2.*sigma**2))
        c2 = (1-(ns/(ns+nb))) * (i**-4.5 * (1/(2.492e-8)))
        y += np.log(c1 + c2)
    return y

def lnLz(m, data0, data1, data2, data3, data4, data5, data6):
    return lnL0(m, data0) + lnL1(m, data1) + lnL2(m, data2) + lnL3(m, data3) + lnL4(m, data4) + lnL5(m, data5) + lnL6(m, data6)

# Load data from files
data0 = np.loadtxt('mgg_cms2020_cat0.txt')
data1 = np.loadtxt('mgg_cms2020_cat1.txt')
data2 = np.loadtxt('mgg_cms2020_cat2.txt')
data3 = np.loadtxt('mgg_cms2020_cat3.txt')
data4 = np.loadtxt('mgg_cms2020_cat4.txt')
data5 = np.loadtxt('mgg_cms2020_cat5.txt')
data6 = np.loadtxt('mgg_cms2020_cat6.txt')

a = 120
b = 132

# Optimize the functions
res0 = minimize_scalar(lambda x: -lnL0(x, data0), bounds=(a, b), method='bounded')
res1 = minimize_scalar(lambda x: -lnL1(x, data1), bounds=(a, b), method='bounded')
res2 = minimize_scalar(lambda x: -lnL2(x, data2), bounds=(a, b), method='bounded')
res3 = minimize_scalar(lambda x: -lnL3(x, data3), bounds=(a, b), method='bounded')
res4 = minimize_scalar(lambda x: -lnL4(x, data4), bounds=(a, b), method='bounded')
res5 = minimize_scalar(lambda x: -lnL5(x, data5), bounds=(a, b), method='bounded')
res6 = minimize_scalar(lambda x: -lnL6(x, data6), bounds=(a, b), method='bounded')
resz = minimize_scalar(lambda x: -lnLz(x, data0, data1, data2, data3, data4, data5, data6), bounds=(a, b), method='bounded')

m0max, lnLm0 = res0.x, -res0.fun
m1max, lnLm1 = res1.x, -res1.fun
m2max, lnLm2 = res2.x, -res2.fun
m3max, lnLm3 = res3.x, -res3.fun
m4max, lnLm4 = res4.x, -res4.fun
m5max, lnLm5 = res5.x, -res5.fun
m6max, lnLm6 = res6.x, -res6.fun
mzmax, lnLmz = resz.x, -resz.fun

m0 = np.arange(124.67, 126.74, 0.001)
m1 = np.arange(124.73, 126.11, 0.001)
m2 = np.arange(123.63, 126.66, 0.001)
m3 = np.arange(123.127, 126.713, 0.001)
m4 = np.arange(126.05, 130.47, 0.001)
m5 = np.arange(123.45, 130.63, 0.001)
m6 = np.arange(123.17, 127.44, 0.001)
mz = np.arange(125.06, 126.02, 0.01)

LnL0 = 2 * (lnL0(m0max, data0) - lnL0(m0, data0))
LnL1 = 2 * (lnL1(m1max, data1) - lnL1(m1, data1))
LnL2 = 2 * (lnL2(m2max, data2) - lnL2(m2, data2))
LnL3 = 2 * (lnL3(m3max, data3) - lnL3(m3, data3))
LnL4 = 2 * (lnL4(m4max, data4) - lnL4(m4, data4))
LnL5 = 2 * (lnL5(m5max, data5) - lnL5(m5, data5))
LnL6 = 2 * (lnL6(m6max, data6) - lnL6(m6, data6))
LnLz = 2 * (lnLz(mzmax, data0, data1, data2, data3, data4, data5, data6) - lnLz(mz, data0, data1, data2, data3, data4, data5, data6))

# Function to plot and save individual plots
def plot_and_save(m, LnL, label, filename):
    plt.figure()
    plt.plot(m, LnL, label=label)
    plt.xlabel('m')
    plt.ylabel('-2 ln L')
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()

plot_and_save(mz, LnLz, 'LnLz', 'LnLz.png')
plot_and_save(m0, LnL0, 'LnL0', 'LnL0.png')
plot_and_save(m1, LnL1, 'LnL1', 'LnL1.png')
plot_and_save(m2, LnL2, 'LnL2', 'LnL2.png')
plot_and_save(m3, LnL3, 'LnL3', 'LnL3.png')
plot_and_save(m4, LnL4, 'LnL4', 'LnL4.png')
plot_and_save(m5, LnL5, 'LnL5', 'LnL5.png')
plot_and_save(m6, LnL6, 'LnL6', 'LnL6.png')

print("m0max:", m0max)
print("m1max:", m1max)
print("m2max:", m2max)
print("m3max:", m3max)
print("m4max:", m4max)
print("m5max:", m5max)
print("m6max:", m6max)
print("mzmax:", mzmax)
def confidence_interval(m_array, LnL_array, max_LnL, threshold):
    # Find the indices where LnL drops below the threshold
    lower_index = np.argmax(LnL_array < (max_LnL - threshold))
    upper_index = np.argmax(LnL_array[::-1] < (max_LnL - threshold))

    # Convert upper index to the actual index in the array
    upper_index = len(LnL_array) - upper_index - 1

    # Get the corresponding m values
    lower_m = m_array[lower_index]
    upper_m = m_array[upper_index]

    return lower_m, upper_m

# Compute confidence intervals for each LnL curve
lower_m0, upper_m0 = confidence_interval(m0, LnL0, np.max(LnL0), 1)
lower_m1, upper_m1 = confidence_interval(m1, LnL1, np.max(LnL1), 1)
lower_m2, upper_m2 = confidence_interval(m2, LnL2, np.max(LnL2), 1)
lower_m3, upper_m3 = confidence_interval(m3, LnL3, np.max(LnL3), 1)
lower_m4, upper_m4 = confidence_interval(m4, LnL4, np.max(LnL4), 1)
lower_m5, upper_m5 = confidence_interval(m5, LnL5, np.max(LnL5), 1)
lower_m6, upper_m6 = confidence_interval(m6, LnL6, np.max(LnL6), 1)
lower_mz, upper_mz = confidence_interval(mz, LnLz, np.max(LnLz), 1)

# Print confidence intervals
print("68.3% Confidence Intervals:")
print("LnL0:", lower_m0, "-", upper_m0)
print("LnL1:", lower_m1, "-", upper_m1)
print("LnL2:", lower_m2, "-", upper_m2)
print("LnL3:", lower_m3, "-", upper_m3)
print("LnL4:", lower_m4, "-", upper_m4)
print("LnL5:", lower_m5, "-", upper_m5)
print("LnL6:", lower_m6, "-", upper_m6)
print("LnLz:", lower_mz, "-", upper_mz)