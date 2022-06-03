#plots and fits a DOUBLE gaussian to gamma ray data
#gamma ray sources are from CO-60
#this specifically is to find the peaks so a conversion factor can be found
#the conversion is taking channel number of the multichannel anylyser
#and converting it to a energy unit
#with the conversion factor we can then find the peak evergy levels for other gamma
#ray sources

from pylab import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
#def gauss(x, H, A, x0, sigma):
#	return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
#channel,counts = np.loadtxt("co-60_3-22-3.asc", usecols=(0,1), skiprows = 1, skiprows = 1)


counts = np.loadtxt("co-60_3-22-1.asc", usecols=(1), skiprows = 600)
channel = np.loadtxt("co-60_3-22-1.asc", usecols=(0), skiprows = 600)


#define the function to be fitted to the data
def double_gaussian(x, params ):
	(A, mu, sigma, A1, mu1, sigma1,B,t) = params
	#y = A*np.exp(-(x-mu)**2/(2*sigma**2))
	y = ((A*np.exp(-(x-mu)**2/(2*sigma**2))) \
    	+(A1*np.exp(-(x-mu1)**2/(2*sigma1**2)))) \
    	+B*exp(-x/t)
	return y


#define that we do two peaks
def double_gaussian_fit( params ):
	fit = double_gaussian( channel, params )
	return (fit - y_proc)

y_proc = np.copy(counts)
y_proc[y_proc < 5] = 0.0

#-----------------------------------below are the initial guesses for the
#------------------------------------A,mu,sigma,A1,mu1,sigma1
fit = leastsq( double_gaussian_fit, [800,770,20,660,870,20,400,35] )

print("fit1",fit)

#plot( channel, counts, c='k' )#ploting raw data
#plot( channel, double_gaussian( channel, fit[0] ), c='b' )#plotting fit

#the fit is being hurt by the fact that there is compton scattering that is bleeding
#into the two peaks and this is throwing off the data

#gonna fit each of the calibration runs and then average the mean
#although i recall something in taylor saying to do something fancy when
#averaging averages

#--------------------------------------------------------------------------------------------------------------------------
counts2 = np.loadtxt("co-60_3-22-2.asc", usecols=(1), skiprows = 600)
channel2 = np.loadtxt("co-60_3-22-2.asc", usecols=(0), skiprows = 600)


#define the function to be fitted to the data
def double_gaussian2(x, params ):
	(A, mu, sigma, A1, mu1, sigma1, B, t) = params
	#y = A*np.exp(-(x-mu)**2/(2*sigma**2))
	y = (A*np.exp(-(x-mu)**2/(2*sigma**2))) \
    	+(A1*np.exp(-(x-mu1)**2/(2*sigma1**2))) \
    	+B*exp(-x/t)
	return y


#define that we do two peaks
def double_gaussian_fit2( params ):
	fit2 = double_gaussian2( channel2, params )
	return (fit2 - y_proc2)

y_proc2 = np.copy(counts2)
y_proc2[y_proc2 < 5] = 0.0

#-----------------------------------below are the initial guesses for the
#------------------------------------A,mu,sigma,A1,mu1,sigma1
fit2 = leastsq( double_gaussian_fit2, [800,770,20,660,870,20,400,35] )

print("fit2",fit2)

#--------------------------------------------------------------------------------------------------------------------------
counts3 = np.loadtxt("co-60_3-22-3.asc", usecols=(1), skiprows = 600)
channel3 = np.loadtxt("co-60_3-22-3.asc", usecols=(0), skiprows = 600)


#define the function to be fitted to the data
def double_gaussian3(x, params ):
	(A, mu, sigma, A1, mu1, sigma1, B, t) = params
	#y = A*np.exp(-(x-mu)**2/(2*sigma**2))
	y = (A*np.exp(-(x-mu)**2/(2*sigma**2))) \
    	+(A1*np.exp(-(x-mu1)**2/(2*sigma1**2))) \
    	+B*exp(-x/t)
	return y


#define that we do two peaks
def double_gaussian_fit3( params ):
	fit3 = double_gaussian3( channel3, params )
	return (fit3 - y_proc3)

y_proc3 = np.copy(counts3)
y_proc3[y_proc3 < 5] = 0.0

#-----------------------------------below are the initial guesses for the
#------------------------------------A,mu,sigma,A1,mu1,sigma1
fit3 = leastsq( double_gaussian_fit3, [800,770,20,660,870,20,400,35] )

print("fit3",fit3)

#--------------------------------------------------------------------------------------------------------------------------
counts5 = np.loadtxt("co-60_3-22-5.asc", usecols=(1), skiprows = 600)
channel5 = np.loadtxt("co-60_3-22-5.asc", usecols=(0), skiprows = 600)


#define the function to be fitted to the data
def double_gaussian5(x, params ):
	(A, mu, sigma, A1, mu1, sigma1, B, t) = params
	#y = A*np.exp(-(x-mu)**2/(2*sigma**2))
	y = (A*np.exp(-(x-mu)**2/(2*sigma**2))) \
    	+(A1*np.exp(-(x-mu1)**2/(2*sigma1**2))) \
    	+B*exp(-x/t)
	return y


#define that we do two peaks
def double_gaussian_fit5( params ):
	fit5 = double_gaussian5( channel5, params )
	return (fit5 - y_proc5)

y_proc5 = np.copy(counts5)
y_proc5[y_proc5 < 5] = 0.0

#-----------------------------------below are the initial guesses for the
#------------------------------------A,mu,sigma,A1,mu1,sigma1
fit5 = leastsq( double_gaussian_fit5, [800,770,20,660,870,20,400,35] )

print("fit5",fit5)

#--------------------------------------------------------------------------------------------------------------------------
counts6 = np.loadtxt("co-60_3-22-6.asc", usecols=(1), skiprows = 600)
channel6 = np.loadtxt("co-60_3-22-6.asc", usecols=(0), skiprows = 600)


#define the function to be fitted to the data
def double_gaussian6(x, params ):
	(A, mu, sigma, A1, mu1, sigma1, B, t) = params
	#y = A*np.exp(-(x-mu)**2/(2*sigma**2))
	y = (A*np.exp(-(x-mu)**2/(2*sigma**2))) \
    	+(A1*np.exp(-(x-mu1)**2/(2*sigma1**2))) \
    	+B*exp(-x/t)
	return y


#define that we do two peaks
def double_gaussian_fit6( params ):
	fit6 = double_gaussian6( channel6, params )
	return (fit6 - y_proc6)

y_proc6 = np.copy(counts6)
y_proc6[y_proc6 < 5] = 0.0

#-----------------------------------below are the initial guesses for the
#------------------------------------A,mu,sigma,A1,mu1,sigma1
fit6 = leastsq( double_gaussian_fit6, [800,770,20,660,870,20,400,35] )

print("fit6",fit6)

#--------------------------------------------------------------------------------------------------------------------------
counts7 = np.loadtxt("co-60_3-22-7.asc", usecols=(1), skiprows = 600)
channel7 = np.loadtxt("co-60_3-22-7.asc", usecols=(0), skiprows = 600)


#define the function to be fitted to the data
def double_gaussian7(x, params ):
	(A, mu, sigma, A1, mu1, sigma1, B, t) = params
	#y = A*np.exp(-(x-mu)**2/(2*sigma**2))
	y = (A*np.exp(-(x-mu)**2/(2*sigma**2))) \
    	+(A1*np.exp(-(x-mu1)**2/(2*sigma1**2))) \
    	+B*exp(-x/t)
	return y


#define that we do two peaks
def double_gaussian_fit7( params ):
	fit7 = double_gaussian7( channel7, params )
	return (fit7 - y_proc7)

y_proc7 = np.copy(counts7)
y_proc7[y_proc7 < 5] = 0.0

#-----------------------------------below are the initial guesses for the
#------------------------------------A,mu,sigma,A1,mu1,sigma1
fit7 = leastsq( double_gaussian_fit7, [800,770,20,660,870,20,400,40] )

print("fit7",fit7)

#--------------------------------------------------------------------------------------------------------------------------
counts8 = np.loadtxt("co-60_3-22-8.asc", usecols=(1), skiprows = 600)
channel8 = np.loadtxt("co-60_3-22-8.asc", usecols=(0), skiprows = 600)


#define the function to be fitted to the data
def double_gaussian8(x, params ):
	(A, mu, sigma, A1, mu1, sigma1, B, t) = params
	#y = A*np.exp(-(x-mu)**2/(2*sigma**2))
	y = (A*np.exp(-(x-mu)**2/(2*sigma**2))) \
    	+(A1*np.exp(-(x-mu1)**2/(2*sigma1**2))) \
    	+B*exp(-x/t)
	return y


#define that we do two peaks
def double_gaussian_fit8( params ):
	fit8 = double_gaussian8( channel8, params )
	return (fit8 - y_proc8)

y_proc8 = np.copy(counts8)
y_proc8[y_proc8 < 5] = 0.0

#-----------------------------------below are the initial guesses for the
#------------------------------------A,mu,sigma,A1,mu1,sigma1
fit8 = leastsq( double_gaussian_fit8, [800,770,20,660,870,20,400,35] )

print("fit8",fit8)

#--------------------------------------------------------------------------------------------------------------------------
counts9 = np.loadtxt("co-60_3-22-9.asc", usecols=(1), skiprows = 600)
channel9 = np.loadtxt("co-60_3-22-9.asc", usecols=(0), skiprows = 600)


#define the function to be fitted to the data
def double_gaussian9(x, params ):
	(A, mu, sigma, A1, mu1, sigma1, B, t) = params
	#y = A*np.exp(-(x-mu)**2/(2*sigma**2))
	y = (A*np.exp(-(x-mu)**2/(2*sigma**2))) \
    	+(A1*np.exp(-(x-mu1)**2/(2*sigma1**2))) \
    	+B*exp(-x/t)
	return y


#define that we do two peaks
def double_gaussian_fit9( params ):
	fit9 = double_gaussian9( channel9, params )
	return (fit9 - y_proc9)

y_proc9 = np.copy(counts9)
y_proc9[y_proc9 < 5] = 0.0

#-----------------------------------below are the initial guesses for the
#------------------------------------A,mu,sigma,A1,mu1,sigma1
fit9 = leastsq( double_gaussian_fit9, [800,770,20,660,870,20,400,35] )

print("fit9",fit9)

#--------------------------------------------------------------------------------------------------------------------------
counts10 = np.loadtxt("co-60_3-22-10.asc", usecols=(1), skiprows = 600)
channel10 = np.loadtxt("co-60_3-22-10.asc", usecols=(0), skiprows = 600)


#define the function to be fitted to the data
def double_gaussian10(x, params ):
	(A, mu, sigma, A1, mu1, sigma1, B, t) = params
	#y = A*np.exp(-(x-mu)**2/(2*sigma**2))
	y = (A*np.exp(-(x-mu)**2/(2*sigma**2))) \
    	+(A1*np.exp(-(x-mu1)**2/(2*sigma1**2))) \
    	+B*exp(-x/t)
	return y


#define that we do two peaks
def double_gaussian_fit10( params ):
	fit10 = double_gaussian10( channel10, params )
	return (fit10 - y_proc10)

y_proc10 = np.copy(counts10)
y_proc10[y_proc10 < 5] = 0.0

#-----------------------------------below are the initial guesses for the
#------------------------------------A,mu,sigma,A1,mu1,sigma1
fit10 = leastsq( double_gaussian_fit10, [800,770,20,660,870,20,400,40] )

print("fit10",fit10)

#--------------------------------------------------------------------------------------------------------------------------
counts11 = np.loadtxt("co-60_3-22-11.asc", usecols=(1), skiprows = 600)
channel11 = np.loadtxt("co-60_3-22-11.asc", usecols=(0), skiprows = 600)


#define the function to be fitted to the data
def double_gaussian11(x, params ):
	(A, mu, sigma, A1, mu1, sigma1, B, t) = params
	#y = A*np.exp(-(x-mu)**2/(2*sigma**2))
	y = (A*np.exp(-(x-mu)**2/(2*sigma**2))) \
    	+(A1*np.exp(-(x-mu1)**2/(2*sigma1**2))) \
    	+B*exp(-x/t)
	return y

#define that we do two peaks
def double_gaussian_fit11( params ):
	fit11 = double_gaussian( channel11, params )
	return (fit11 - y_proc)

y_proc = np.copy(counts11)
y_proc[y_proc < 5] = 0.0

#-----------------------------------below are the initial guesses for the
#------------------------------------A,mu,sigma,A1,mu1,sigma1
fit11 = leastsq( double_gaussian_fit11, x0 = [800,770,20,660,870,20,400,35])

print("fit11",fit11)

#--------------------------------------------------------------------------------------------------------------------------
counts12 = np.loadtxt("co-60-20-1.asc", usecols=(1), skiprows = 600)
channel12 = np.loadtxt("co-60-20-1.asc", usecols=(0), skiprows = 600)


#define the function to be fitted to the data
def double_gaussian12(x, params ):
	(A, mu, sigma, A1, mu1, sigma1, B, t) = params
	#y = A*np.exp(-(x-mu)**2/(2*sigma**2))
	y = (A*np.exp(-(x-mu)**2/(2*sigma**2))) \
    	+(A1*np.exp(-(x-mu1)**2/(2*sigma1**2))) \
    	+B*exp(-x/t)
	return y

#define that we do two peaks
def double_gaussian_fit12( params ):
	fit12 = double_gaussian( channel12, params )
	return (fit12 - y_proc)

y_proc = np.copy(counts12)
y_proc[y_proc < 5] = 0.0

#-----------------------------------below are the initial guesses for the
#------------------------------------A,mu,sigma,A1,mu1,sigma1
fit12 = leastsq( double_gaussian_fit12, x0 = [800,770,20,660,870,20,400,35])

print("fit12",fit12)

#--------------------------------------------------------------------------------------------------------------------------
counts13 = np.loadtxt("co-60-20-2.asc", usecols=(1), skiprows = 600)
channel13 = np.loadtxt("co-60-20-2.asc", usecols=(0), skiprows = 600)


#define the function to be fitted to the data
def double_gaussian13(x, params ):
	(A, mu, sigma, A1, mu1, sigma1, B, t) = params
	#y = A*np.exp(-(x-mu)**2/(2*sigma**2))
	y = (A*np.exp(-(x-mu)**2/(2*sigma**2))) \
    	+(A1*np.exp(-(x-mu1)**2/(2*sigma1**2))) \
    	+B*exp(-x/t)
	return y

#define that we do two peaks
def double_gaussian_fit13( params ):
	fit13 = double_gaussian( channel13, params )
	return (fit13 - y_proc)

y_proc = np.copy(counts13)
y_proc[y_proc < 5] = 0.0

#-----------------------------------below are the initial guesses for the
#------------------------------------A,mu,sigma,A1,mu1,sigma1
fit13 = leastsq( double_gaussian_fit13, x0 = [800,770,20,660,870,20,400,35])

print("fit13",fit13)

plt.subplot(6, 2, 1)
plt.plot(channel, counts, c = 'r')
plt.plot(channel, double_gaussian( channel, fit[0] ), c = 'b')


plt.subplot(6, 2, 2)
plt.plot(channel2, counts2, c = 'r')
plt.plot(channel2, double_gaussian2( channel2, fit2[0] ), c = 'b')


plt.subplot(6, 2, 3)
plt.plot(channel3, counts3, c = 'r')
plt.plot(channel3, double_gaussian3( channel3, fit3[0] ), c = 'b')


plt.subplot(6, 2, 4)
plt.plot(channel5, counts5, c = 'r')
plt.plot(channel5, double_gaussian5( channel5, fit5[0] ), c = 'b')


plt.subplot(6, 2, 5)
plt.plot(channel6, counts6, c = 'r')
plt.plot(channel6, double_gaussian6( channel6, fit6[0] ), c = 'b')


plt.subplot(6, 2, 6)
plt.plot(channel7, counts7, c = 'r')
plt.plot(channel7, double_gaussian7( channel7, fit7[0] ), c = 'b')


plt.subplot(6, 2, 7)
plt.plot(channel8, counts8, c = 'r')
plt.plot(channel8, double_gaussian8( channel8, fit8[0] ), c = 'b')


plt.subplot(6, 2, 8)
plt.plot(channel9, counts9, c = 'r')
plt.plot(channel9, double_gaussian9( channel9, fit9[0] ), c = 'b')


plt.subplot(6, 2, 9)
plt.plot(channel10, counts10, c = 'r')
plt.plot(channel10, double_gaussian10( channel10, fit10[0] ), c = 'b')


plt.subplot(6, 2, 10)
plt.plot(channel11, counts11, c = 'r')
plt.plot(channel11, double_gaussian11( channel11, fit11[0] ), c = 'b')

plt.subplot(6, 2, 11)
plt.plot(channel12, counts12, c = 'r')
plt.plot(channel12, double_gaussian12( channel12, fit12[0] ), c = 'b')

plt.subplot(6, 2, 12)
plt.plot(channel13, counts13, c = 'r')
plt.plot(channel13, double_gaussian13( channel13, fit13[0] ), c = 'b')

plt.suptitle("Co-60")
plt.show()


#plotting all the fits and data


#after putting the mu and mu1 values in a txt file we will now
#take those values to find deltaMU, and then we can find the
#conversion factor between channel number and energy

mu = np.loadtxt("mu_values.txt", usecols=(0), skiprows = 1)
mu1 = np.loadtxt("mu_values.txt", usecols=(1), skiprows = 1)

peak1_energy = 1173.2 #kev
peak2_energy = 1332.5 #kev

deltaE = 159.3 #kev

mu_average = np.average(mu)
mu1_average = np.average(mu1)

delta_mu = mu1-mu
print(delta_mu)

conversion_factor = deltaE/delta_mu
conversion_factor_avg = np.average(conversion_factor)

print("Average conversion factor from channel to kev = ", conversion_factor_avg)

sigma_con = np.std(conversion_factor)
print("sigma in conversion factor: ", sigma_con)

convert = conversion_factor*mu_average
convert1 = conversion_factor*mu1_average
#print(convert)
#print(convert1)

#conversion is not perfect due to compton scattering systematic error

#plotting one of the fits for the report
plot( channel12, counts12, c='r' )#ploting raw data
plot( channel12, double_gaussian12( channel12, fit12[0] ), c='b' )#plotting fit
xlabel ('Channel Number')
ylabel ('Counts')
title ('Co-60, run 12, with double peak fit')


sigma1 = np.loadtxt("sigma.txt", usecols=(0), skiprows = 1)
sigma2 = np.loadtxt("sigma.txt", usecols=(1), skiprows = 1)

sigma1_avg = np.average(sigma1)
sigma2_avg = np.average(sigma2)

#print(sigma1_avg)
#print(sigma2_avg)


