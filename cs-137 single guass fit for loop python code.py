#SINGLE GUASS FIT

#plots and fits a gaussian to gamma ray data
#this specifically is to find the peaks so a conversion factor can be found
#with the conversion factor we can then find the peak evergy levels for other gamma
#ray sources

from pylab import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math
from astropy.io import ascii
from astropy.table import Table, Column
import os
from openpyxl import load_workbook
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.io.fits as pyfits

from numpy import rec
import random
import pandas
import pandas as pd
import seaborn as sns
import math 
from pylab import *
from math import factorial
from decimal import Decimal
from sympy import multiplicity
import scipy.special
from scipy.optimize import curve_fit
#def gauss(x, H, A, x0, sigma):
#	return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def Gauss(x, A, mu, sigma):
	return A*np.exp(-(x-mu)**2/(2*sigma**2))

test_number = 48 # Number of runs plus 1

i = 1

for i in range(1,test_number):
    counts = np.loadtxt("CS-137-" + str(i) + ".asc", usecols=(1), skiprows = 375, max_rows = 150)
    channel = np.loadtxt("CS-137-" + str(i) + ".asc", usecols=(0), skiprows = 375, max_rows = 150)
    
    parameters, covariance = curve_fit(Gauss, channel, counts, p0=[1000,450,50])
    cerr = np.sqrt(np.diag(covariance))

    fit_A = parameters[0]
    fit_mu = parameters[1]
    fit_sigma = parameters[2]

    fit_y = Gauss(channel, fit_A, fit_mu, fit_sigma)
    #print(fit_A)
    #print(fit_mu)
    #print(fit_sigma)
    
    plt.subplot(1, 1, 1)
    plt.plot(channel, counts, c = 'r')
    plt.plot(channel, fit_y, c = 'b')
    xlabel ('Channel Number')
    ylabel ('Counts')
    title ('Cs-137, Spectra '+ str(i))
    plt.savefig('Cs-137, spectra '+ str(i)+ ".png")
    plt.xlim(360, 540)
    plt.ylim(-50,1200)
    plt.show()
    i+1
    
    
#taking the mu data from a data file
Cs_137_mu= np.loadtxt("cs-137 mu.txt", usecols=(0), skiprows = 1)

conversion_factor = 1.5713804269120362 #gotten from other program

mu_avg = np.average(Cs_137_mu)

convert_mu = Cs_137_mu*conversion_factor

#print("Peak locations: ", convert_mu, "kev")
mu_avg = np.average(convert_mu)
print(mu_avg, "kev")
peak_expect = 662 #kev

percent_error = ((peak_expect-convert_mu)/convert_mu)*100

#print(abs(percent_error))

percent_error_tot = ((peak_expect-mu_avg)/mu_avg)*100
print("Percent error: ", abs(percent_error_tot), "%")

sigma = np.loadtxt("cs-137 sigmas.txt", usecols=(0), skiprows = 1)
convert_sigma = sigma*conversion_factor
#print(convert_sigma)
sigma_average = np.average(convert_sigma)
print("The average error in mu: +/-", sigma_average, "kev")


#finding wavelength of photopeak
h = 4.1357*10**(-12) #kev
c = 2.9979*10**8 #m/s
lamda  = (h*c)/(Cs_137_mu*conversion_factor)

print("wavelength values: ", lamda*10**9)

lamda_avg = np.average(lamda)
print("average wavelength: ", lamda_avg*10**9)

sigma_lamda = np.std(lamda)
print(sigma_lamda*10**9)



    