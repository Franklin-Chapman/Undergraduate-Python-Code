#Astrophys hw2 problem 2
#program computes flux values for G2V and O5V type stars
#then plots the flux of the two stars as a function of distance
#it has a log scale y-axis, and a linear scale on the x-axis
 
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math
import dill as pickleout
from astropy.io import ascii
from astropy.table import Table, Column
 
 
#read in raw data file(G2V type)
distance,flux=loadtxt('G2V.dat',unpack=True,skiprows=1,usecols=(0,1))
 
#calculate flux values
#distance values from 1au to 1pc in cm
r=np.array([1.5*10**13,3.1*10**17,6.2*10**17,9.3*10**17,1.2*10**18,1.5*10**18,1.9*10**18,2.2*10**18,2.5*10**18,2.8*10**18,3.1*10**18])
 
#send array through formulas
#G2V type luminosity = 3.83*10^33
f1=3.83*10**33/(4*pi*r**2)
#O5V type
f2=4.08*10**35/(4*pi*r**2)
print(f1)
#put flux data into an array
flux1=[]
flux1.append(f1)
#print(flux1)
 
#store PC distance to dat file
#PC distance array
rpc=np.array([0.000005,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
 
   
t1=ascii.write([rpc,f1, r], 'G2V.dat',names=['-Dist-pc','-Flux', '-Dist-cm'],formats={'-Dist-cm':'2.0f','-Dist-pc':'.2f','Flux':'.1f'})
t2=ascii.write([rpc,f2, r], 'O5V.dat',names=['-Dist-pc','-Flux', '-Dist-cm'],formats={'-Dist-cm':'2.0f','-Dist-pc':'.2f','Flux':'.1f'})
 
# plot the data Flux vs distance(pc)(G2V)
plt.plot(distance,flux,'gx',label='My Plot')
plt.plot(distance,flux,'b')
#general plot details or both stars
plt.xlabel('Distance(pc)')
plt.ylabel('Flux(erg*s^-1*cm^-2)(on log scale)')
plt.xlim([0.000005,1.1]) # set range on axis
plt.plot(distance, flux)
plt.yscale('log')
plt.grid(True)
plt.title('Flux VS. Distance')
 
 
 
 
#O5V type luminosity = 4.08*10^35
f2=4.08*10**35/(4*pi*r**2)
#print(f2)
 
 
 
#legend
red_patch = mpatches.Patch(color='red', label='O5V type')
plt.legend(handles=[red_patch])
plt.legend(handles=[], loc='lower right')
 
#read in raw data file(O5V type)
distance,flux=loadtxt('O5V.dat',unpack=True,skiprows=1,usecols=(0,1))
 
# plot the data Flux vs distance(pc)(O5V)
plt.plot(distance,flux,'gx',label='My Plot')
plt.plot(distance,flux,'r')
#plt.xlabel('Distance(pc)')
#plt.ylabel('Flux(erg*s^-1*cm^-2')
#axis([0.000005,1.1,0.004,-0.001]) # set range on axis
#plt.annotate('Blaa',(0.8,2))
