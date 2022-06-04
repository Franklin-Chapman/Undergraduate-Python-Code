#program imports txt file and fits a multi gauss to it and does data analysis
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.optimize import leastsq
from scipy.optimize import curve_fit

#defining the multi_gauss for the 6 peaked large Franck Hertz experiment
def multi_gauss(x, A, mu, sigma, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3, A4,mu4,sigma4,A5,mu5,sigma5 ):
    y = (A*np.exp(-(x-mu)**2/(2*sigma**2))) \
              +(A1*np.exp(-(x-mu1)**2/(2*sigma1**2))) \
                  +(A2*np.exp(-(x-mu2)**2/(2*sigma2**2))) \
                      +(A3*np.exp(-(x-mu3)**2/(2*sigma3**2))) \
                          +(A4*np.exp(-(x-mu4)**2/(2*sigma4**2))) \
                              +(A5*np.exp(-(x-mu5)**2/(2*sigma5**2)))
    return y



test_number = 5 # Number of runs plus 1

i = 1
#empty list to fill in for loop with fit parameters
l_mu1 = []
l_mu2 = []
l_mu3 = []
l_mu4 = []
l_mu5 = []
l_mu6 = []


l_sigma1 = []
l_sigma2 = []
l_sigma3 = []
l_sigma4 = []
l_sigma5 = []
l_sigma6 = []


for i in range(1,test_number):
    #open file
    #"neon " + str(i) + ".txt"
    v_accel = np.loadtxt("largon " + str(i) + ".txt", usecols=(1), skiprows = 1)
    i_ampli = np.loadtxt("largon " + str(i) + ".txt", usecols=(0), skiprows = 1)

    #define fit function 
    parameters, covariance = curve_fit(multi_gauss, v_accel, i_ampli, maxfev=100000, 
                                       p0=[100,35,5,140,45,5,200,55,5,300,67,5,350,80,5,440,93,5])
    #error in fit
    convariance_error = np.sqrt(np.diag(covariance))
    
    #fit parameters to go into empty list above
    fit_mu = parameters[1]
    fit_sigma = parameters[2]
    fit_mu2 = parameters[4]
    fit_sigma2 = parameters[5]
    fit_mu3 = parameters[7]
    fit_sigma3 = parameters[8]
    fit_mu4 = parameters[10]
    fit_mu5 = parameters[13]
    fit_mu6 = parameters[16]
    fit_sigma4 = parameters[11]
    fit_sigma5 = parameters[14]
    fit_sigma6 = parameters[17]
    
    #filling list with values as the loop loops
    l_mu1.append(fit_mu)
    l_mu2.append(fit_mu2)
    l_mu3.append(fit_mu3)
    l_mu4.append(fit_mu4)
    l_mu5.append(fit_mu5)
    l_mu6.append(fit_mu6)
    l_sigma1.append(fit_sigma)
    l_sigma2.append(fit_sigma2)
    l_sigma3.append(fit_sigma3)
    l_sigma1.append(fit_sigma4)
    l_sigma2.append(fit_sigma5)
    l_sigma3.append(fit_sigma6)
    
    #fit happens
    fit_y =  multi_gauss(v_accel, *parameters)
    
    #plotting the final results
    
    # plt.subplot(5, 2, i)
    # suptitle(' Neon ', fontsize=20)
    # tight_layout()
    plt.subplot(1, 1, 1)
    plt.plot(v_accel, i_ampli, 'ro',markersize=2,label='Raw Data')
    plt.plot(v_accel, fit_y, 'b',markersize=2,label='Gauss Fit')
    # xlabel ('Voltage(V)')
    # ylabel ('Current(A)')
    title ('Argon L Franck Hertz '+ str(i))
    # legend()
    plt.savefig('Argon L'+ str(i)+ ".png")    
    plt.show()
    i+1
# text(-13, -11, 'Voltage (V)', ha='center')  
# text(-140, 25, 'Current (A)', va='center', rotation='vertical')
# plt.savefig('Neon  subplot'+ str(i)+ ".png")    
# plt.show()



#analysis of peak locations
#converting list to array for analysis
mu1 = np.asarray(l_mu1)
mu2 = np.asarray(l_mu2)
mu3 = np.asarray(l_mu3)
mu4 = np.asarray(l_mu4)
mu5 = np.asarray(l_mu5)
mu6 = np.asarray(l_mu6)


sigma1 = np.asarray(l_sigma1)
sigma2 = np.asarray(l_sigma2)
sigma3 = np.asarray(l_sigma3)
sigma4 = np.asarray(l_sigma4)
sigma5 = np.asarray(l_sigma5)
sigma6 = np.asarray(l_sigma6)


#finding difference between peaks
dif1 = mu2-mu1
print("Excitation energy 1: ",dif1)
dif2 = mu3-mu2
print("Excitation energy 2: ",dif2)
dif3 = mu4-mu3
print("Excitation energy 3: ",dif3)
dif4 = mu5-mu4
print("Excitation energy 4: ",dif4)
dif5 = mu6-mu5
print("Excitation energy 5: ",dif5)


#totalling up differences 
dif = np.array([dif1,dif2,dif3,dif4,dif5])
#taking average
average_excite = np.average(dif)
print(average_excite, "eV")

