#test fitting a bunch of Gauss to the small amount of data I got from the good Franck Hertz setup
#before it was broken...

from pylab import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.optimize import leastsq
from scipy.optimize import curve_fit


# def multi_gauss(x, a0,A, mu, sigma, a1,A1, mu1, sigma1, a2,A2, mu2, sigma2, a3,A3, mu3, sigma3
#             	,a4,A4, mu4, sigma4,a5,A5, mu5, sigma5,a6,A6, mu6, sigma6,a7,A7, mu7,
#             	sigma7,a8,A8, mu8, sigma8 ):
# 	y = (a0+A*np.exp(-(x-mu)**2/(2*sigma**2))) \
#           	+(a1+A1*np.exp(-(x-mu1)**2/(2*sigma1**2))) \
#               	+(a2+A2*np.exp(-(x-mu2)**2/(2*sigma2**2))) \
#                   	+(a3+A3*np.exp(-(x-mu3)**2/(2*sigma3**2))) \
#                       	+(a4+A4*np.exp(-(x-mu4)**2/(2*sigma4**2))) \
#                           	+(a5+A5*np.exp(-(x-mu5)**2/(2*sigma5**2))) \
#                               	+(a6+A6*np.exp(-(x-mu6)**2/(2*sigma6**2))) \
#                                   	+(a7+A7*np.exp(-(x-mu7)**2/(2*sigma7**2))) \
#                                       	+(a8+A8*np.exp(-(x-mu8)**2/(2*sigma8**2)))
# 	return y
def s1tos2(t,A0,b,omega,phi,S,D ):
	y = A0*np.exp(-b/(2*76.6)*t)*np.cos(omega*t-phi)+(S*t+D)
	return y

def s2tos1(t, A01,b1,omega1,phi1,S1,D1 ):
	y = A01*np.exp(-b1/(2*76.6)*t)*np.cos(omega1*t-phi1)+(S1*t+D1)
	return y



test_number = 2 # Number of runs plus 1

i = 1

T = []
T1 = []
for i in range(1,test_number):

	v_accel = np.loadtxt("sn8.txt", usecols=(0), skiprows = 2, max_rows = 95)
	i_ampli = np.loadtxt("sn8.txt", usecols=(1), skiprows = 2, max_rows = 95)
	sigmay = 0*v_accel+0.01 #cm
	#print(i_ampli)
	parameters, covariance = curve_fit(s1tos2, v_accel, i_ampli, sigma = sigmay, absolute_sigma = True,  maxfev=10000000,
                                   	p0=[4.2986508,3.9223091,0.5364628,13.329883,0,0.0031964])
	cerr = np.sqrt(np.diag(covariance))

	fit_omega = parameters[2]
	S = parameters[4]
	D = parameters[5]
	fit_T = (2*np.pi)/fit_omega   
	T.append(fit_T)

    

	v_accel1 = np.loadtxt("sn8.txt", usecols=(0), skiprows = 98)#, max_rows = 60)
	i_ampli1 = np.loadtxt("sn8.txt", usecols=(1), skiprows = 98)#, max_rows = 60)
	sigmay1 = 0*v_accel+0.01 #cm
	parameters1, covariance1 = curve_fit(s2tos1, v_accel1, i_ampli1,  maxfev=10000000,
                                    	p0=[144.2473,10.40035,0.467602,0,0.026661,0.223])
	cerr = np.sqrt(np.diag(covariance1))

	fit_y =  s1tos2(v_accel, *parameters)
	fit_y1 = s2tos1(v_accel1, *parameters1)

	fit_omega1 = parameters1[2]
	S1 = parameters1[4]
	D1 = parameters1[5]
	fit_T1 = (2*np.pi)/fit_omega1   
	T1.append(fit_T1)
    
	#plotting
	plt.subplot(1, 1, 1)
	plt.plot(v_accel, i_ampli, 'r+',markersize=3,label='Raw Data 1')
	#plt.plot(v_accel1, i_ampli1, 'r+',markersize=3,label='Raw Data 2')
	plt.plot(v_accel, fit_y, 'b',markersize=5,label="Fit 1")
	#plt.plot(v_accel1, fit_y1, 'b',markersize=5,label="Fit 2")
	plt.plot(v_accel, S*v_accel+D, 'g', markersize = 5)
	#plt.plot(v_accel1, S1*v_accel1+D1, 'g', markersize = 5)
	xlabel ('Time(min)')
	ylabel ('Distance(cm)')
    
	s1 = S*59+D
	s2 = S*0+D
	print(s1,s2)

	title ('SN8 ')
	legend()
	plt.savefig('Small Argon'+ str(i)+ ".png")    
	plt.show()
	#print(sigma1)
   	 
	i+1
# text(-7, -0.5, 'Voltage (V)', ha='center')  
# text(-85, 1.3, 'Current (mA)', va='center', rotation='vertical')
# plt.show()


#analysis of peak locations

period = np.asarray(T)


#fingding G
L = 9247.475/1000#cm
s2 = 0.5399691173145514 #cm
#s2 = 0.25/100
s1 = 0.4963615203279986 #cm
r = 9.55/1000 #cm
d = 50/1000 #cm
b = 42.2/1000 #cm
m1 = 1.5 #kg

# T1 = mu2-mu1
# print("Period T1 ",T1)
# T2 = mu3-mu2
# print("Peroid T2 ",T2)
# period = np.array([T1,T2])
# T = np.average(period)
deltaS = s2-s1

G = np.pi**2*(s2-s1)*b**2*((d**2+2/5*r**2)/(period**2*m1*L*d))
print("value for G : ", G)




# dif1 = mu2-mu1
# print("Excitation energy 1: ",dif1)
# dif2 = mu3-mu2
# print("Excitation energy 2: ",dif2)
# dif = np.array([dif1,dif2])
# average_excite = np.average(dif2)
# print(average_excite, "eV")





