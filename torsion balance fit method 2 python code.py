from pylab import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches


def s1tos2(x,A,mu,sigma,A1,mu1,sigma1,D):
	y = (A*np.exp(-(x-mu)**2/(2*sigma**2)))+(A1*np.exp(-(x-mu1)**2/(2*sigma1**2)))+D
	return y

                
                
                
test_number = 2 # Number of runs plus 1

i = 1

T = []
T1 = []
for i in range(1,test_number):
    v_accel = np.loadtxt("torsion run - Sheet1.tsv", usecols=(0), skiprows = 1)#, max_rows = 95)
    i_ampli = np.loadtxt("torsion run - Sheet1.tsv", usecols=(1), skiprows = 1)#, max_rows = 95)
    sigmay = 0*v_accel+0.01 #cm
    print(v_accel)
    parameters, covariance = curve_fit(s1tos2, v_accel, i_ampli, maxfev=10000000,
                                       p0=[190,450,20,187,1000,20,150])
    cerr = np.sqrt(np.diag(covariance))
    
    mu = parameters[1]
    mu1 = parameters[4]
    
# 	fit_omega = parameters[2]
# 	S = parameters[4]
# 	D = parameters[5]
# 	fit_T = (2*np.pi)/fit_omega   
# 	T.append(fit_T)
    fit_y =  s1tos2(v_accel, *parameters)
	#plotting
    plt.subplot(1, 1, 1)
    plt.plot(v_accel, i_ampli, 'r+',markersize=5,label='| G=1.0e-10 +/-2.7e-10 | G0=1.1e-10 +/-2.7e-10 | G%off = 34% | G0%off = 38%')
	#plt.plot(v_accel1, i_ampli1, 'r+',markersize=3,label='Raw Data 2')
    plt.plot(v_accel, fit_y, 'b',markersize=5,label="Fit 1")
	#plt.plot(v_accel1, fit_y1, 'b',markersize=5,label="Fit 2")
	#plt.plot(v_accel, S*v_accel+D, 'g', markersize = 5)
	#plt.plot(v_accel1, S1*v_accel1+D1, 'g', markersize = 5)
    #plt.text(600,185,'This text starts at point (2,4)')
    xlabel ('Time(s)')
    ylabel ('Distance(cm)')
    title("Method 1")
    legend()
    
    
    
    
    T = mu1-mu
    print(T)
    
T = 523.3930879665539 #s
sigma_T = 0.1 #s
#fingding G
L = 9231.6/1000#m 
sigma_L = 0.5 #cm
s2 = 192.9/100 #cm
#s2 = 0.25/100
s1 = 150.1/100
sigma_s = 0.5 #m
r = 9.55/1000 #m
sigma_r = 0.1 #m
d = 50/1000 #m
sigma_d = 0.1 #m
b = 42.2/1000 #m
sigma_b = 0.1 #m
m1 = 1.5 #kg
sigma_m = 0.2 #kg

deltaS = s2-s1

G = np.pi**2*(s2-s1)*b**2*((d**2+2/5*r**2)/(T**2*m1*L*d))
print("value for G : ", G)

#error prop
dd = np.pi**2*deltaS*b**2*((5*d**2-2*r**2)/(5*d**2*T**2*L*m1))
db = 2*np.pi*deltaS*b*((d**2+2/5*r**2)/(T**2*m1*L*d))
dr = 4*np.pi*deltaS*b**2*r*(1/(5*d*T**2*m1*L))
ds = np.pi**2*b**2*((d**2+2/5*r**2)/(T**2*m1*L*d))
dT = -np.pi**2*deltaS*b**2*((d**2+2/5*r**2)/(T**3*m1*L*d))
dL = -np.pi**2*deltaS*b**2*((d**2+2/5*r**2)/(T**2*m1*L**2*d))
dm1 = -np.pi**2*deltaS*b**2*((d**2+2/5*r**2)/(T**2*m1**2*L*d))

sigma_G = np.sqrt(dd**2*sigma_d**2+db**2*sigma_b**2+dr**2*sigma_r**2+ds**2*sigma_s**2+dT**2*sigma_T**2+dL**2*sigma_L**2+dm1**2*sigma_m**2)
print(sigma_G)

#correction for G
beta = (b**3)/((b**2+4*d**2)**(3/2))
G0 = G/(1-beta)
print(G0)


G_ac = 6.67430*10**(-11)

percent_errorG = ((G_ac-G)/G)*100
print(percent_errorG)
percent_errorG0 = ((G_ac-G0)/G0)*100
print(percent_errorG0)
sigma_off = np.abs((G-G_ac))/(sigma_G)
print(sigma_off)
