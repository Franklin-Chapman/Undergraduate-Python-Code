#Program imports in BIAS, DARKS, and FLATS and edits the science images to give us the final science image.
#First the program makes master images of BIAS, DARKS, and FLATS to then be applied to science images
#Second the master images are applied in one for loop
#In addition it displays the master images
#want to add where it selects the first science image and shows the original file, and the final file

import matplotlib.pyplot as plt
import numpy as np
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
import os
import os
import sys
import fnmatch
from astropy import stats
from astropy.io import fits
from tkinter import Tk
from tkinter.filedialog import askdirectory
from collections import namedtuple
import tkinter as tk
from tkinter import Label
from tkinter import *
from tkinter import ttk
from tkinter.messagebox import showinfo

from tkinter import Label
#plt.style.use(astropy_mpl_style)
#file_path = '/d/mic1/franklin/data/TOI_3714/20211119/b-0001.fit'

#UI code to replace the inputs below
#widget = Label(None,text = 'should have worked')
#widget.pack()
#widget.mainloop()

# Defined find function

#------------------------------------------------------------------------------------------------------------------------

def find(pattern, locate):   # locates files with a given pattern just like linux
result = []
for root, dirs, files in os.walk(locate):
for name in files:
if fnmatch.fnmatch(name, pattern):
result.append(name)
return result
#------------------------------------------------------------------------------------------------------------------------


#asking the user for the file path
#we assume parts of the file path is always the same for all users of this program
#-------------------------------------------------------------------------------------------------------------------------

#first we ask the user for their linux user name
#username = input("Linux username: ")
#username check


#second ask for what computer the user is getting the data from (mine is mic1)
#computer = input("Computer data is stored on: ")
#computer check

#third ask for what target the user is analysing
#target = input("Target TOI name: ")
#target check

#lasting the date the data was taken
#date = input("Date data was taken: ")
#date check


#file_name = input("Enter the file path: ")



root = tk.Tk()
root.geometry("700x500")
root.resizable(False,False)
root.title("RBO Reductions")


#variables to store the inputs from the window
username = tk.StringVar()
computer = tk.StringVar()
target = tk.StringVar()
date = tk.StringVar()




#defining the RUN button function
def run_clicked():
""" callback when the run button clicked """
msg = f'You entered username: {username.get()} computer name: {computer.get()} target name: {target.get()} and date {date.get()}'
showinfo(title = 'Imformation', message = msg)



RBO_r = ttk.Frame(root)
RBO_r.pack(padx = 10, pady = 10, fill = 'x', expand = True)


#username_entry
username_label = ttk.Label(RBO_r, text = "Linux Username:")
username_label.pack(fill = 'x', expand = True)
username_entry = ttk.Entry(RBO_r, textvariable = username)
username_entry.pack(fill = 'x', expand = True)
username_entry.focus()

#computer_entry
computer_label = ttk.Label(RBO_r, text = "Computer with data on it:")
computer_label.pack(fill = 'x', expand = True)
computer_entry = ttk.Entry(RBO_r, textvariable = computer)
computer_entry.pack(fill = 'x', expand = True)

#target_entry
target_label = ttk.Label(RBO_r, text = "Target name:")
target_label.pack(fill = 'x', expand = True)
target_entry = ttk.Entry(RBO_r, textvariable = target)
target_entry.pack(fill = 'x', expand = True)

#date_entry
date_label = ttk.Label(RBO_r, text = "Date data was taken:")
date_label.pack(fill = 'x', expand = True)
date_entry = ttk.Entry(RBO_r, textvariable = date)
date_entry.pack(fill = 'x', expand = True)

#run button
run_button = ttk.Button(RBO_r, text = "Run script", command = run_clicked)
run_button.pack(fill = 'x', expand = True, pady = 10)



root.mainloop()





file_name_user = '/d/' + computer + '/' + username + '/data/' + target + '/' + date


print(file_name_user)
file_path_check = input("Does this file path look correct?: ")
N = 1
if file_path_check == N:
which_part_change = input("What part of the path would you like to change?: ") #this really wont work as of now so i'll fix it later
else:
file_name_user = file_name_user

#defining file_name because I don't want to change all the times this has file_name to file_name_user
#file_name = file_name_user
#-------------------------------------------------------------------------------------------------------------------------
file_name = '/d/mic1/franklin/data/TOI_3714/20211119'
#file_name will be provided by the user


#have thing check the dimensions and not ask the user because he does not need to know anything
image_d_check = input("Are the image dimensions still 2048,2048,-1? (Y or N): ")
N = 1
if image_d_check == N:
print("New dimensions format: 'X,Y,what ever the -1 means'")
new_d = input("What are the new dimensions?: ") #this really wont work as of now so i'll fix it later
else:
image_dimensions = (2048,2048,-1)


#MASTER BIAS CREATION
#------------------------------------------------------------------------------------------------------------------------
#using the find function
bias_find = find('b-*.fit', '/d/mic1/franklin/data/TOI_3714/20211119')
number_bias_frames = len(bias_find)

#will be asking user if the dimensions are different this time around.  Prob pointless to ask but idk

#for loop to open each bias and put into a 3D array

for i in range(number_bias_frames):
bias_open = fits.open(file_name + '/' + bias_find[i])
#check to see if we have opened bias images and it kills the program
bias_exposure = bias_open[0].header['exposure']
if bias_exposure != 0:
print("Bias images not found, terminating activity")
quit()
else:

bias_1D = bias_open[0].data.reshape(image_dimensions)
bias_3D = np.array([n for n in bias_1D])
bias_exp_average = np.average(bias_exposure)
print('average bias exposure: ', bias_exp_average, "seconds") #gives zero if the bias images are selected

master_bias = np.median(bias_3D, axis = 2)
print(type(master_bias))
science_find = find('a-*.fit', '/d/mic1/franklin/data/TOI_3714/20211119')
number_science_frames = len(science_find)
#science_data = fits.getdata(file_name + '/' + science_find, ext = 0)

#subtracting bias from science images
for i in range(number_science_frames):
science_import = fits.open('/d/mic1/franklin/data/TOI_3714/20211119/' + science_find[i])
bias_sub = np.subtract(science_import[0].data, master_bias[i])

#BIAS reduction
#science_1D = science_import[0].data.reshape(2048,2048, -1)
#science_3D = np.array([n for n in science_1D])

#print(type(science_3D))
#bias_subtract = np.subtract(science_3D, master_bias[i])
file_name = '*_b.fits'
bias_sub.writeto('/d/mic1/franklin/data/TOI_3714/20211119_new_data/' + file_name)




#plot the masterbias to see what it looks like

#vmax = master_bias*np.std(master_bias)
#vmin = master_bias*np.std(master_bias)



#------------------------------------------------------------------------------------------------------------------------

#MasterDARK creation

#import darks
#dark_find = find('d-*.fit', '/d/mic1/franklin/data/TOI_3714/20211119')
#number_dark_frames = len(dark_find)
#dark_loop_length = range(number_dark_frames)

#for i in dark_loop_length:
# dark_open = fits.open('/d/mic1/franklin/data/TOI_3714/20211119/' + dark_find[i])
# dark_1D = bias_open[0].data.reshape((2048,2048,-1))
# #dark_1D = bias_open[0].data
# dark_3D = np.array([n for n in dark_1D])
#need to add sigma clipping

#master_dark = np.median(dark_3D, axis=2)


#------------------------------------------------------------------------------------------------------------------------

#MasterFLAT creation

#import flats

#flats_find = find('f-*.fit', '/d/mic1/franklin/data/TOI_3714/20211119')




#------------------------------------------------------------------------------------------------------------------------

#science import and subtraction of bias, darks, and flats

#import the science data
#science_find = find('a-*.fit', '/d/mic1/franklin/data/TOI_3714/20211119')
#number_science_frames = len(science_find)
#science_data = fits.getdata(file_name + '/' + science_find, ext = 0)

#print(science_data.shape)
#for i in range(number_science_frames):
# science_open = np.array([fits.open(file_name + '/' + science_find[i])])

#print(len(science_open))
#subtract the masterbias off of the science image and save to a new file
#print("number of science frames: ", number_science_frames)
#for i in range(number_science_frames):
































#for i in range(number_science_frames):
# science_import = fits.open('/d/mic1/franklin/data/TOI_3714/20211119/' + science_find[i])
#BIAS reduction
# bias_subtract = np.subtract(science_import[0].data, master_bias)
# file_name = '*_b.fits'
# fits.writeto('/d/mic1/franklin/data/TOI_3714/20211119/' + file_name, bias_subtract, overwrite=True)
#DARKS reduction


#FLATS reduction

#final science image save
# fits.writeto('/d/mic1/franklin/data/TOI_3714/20211119/' + file_name, bias_subtract, overwrite=True)
# science_import.close()

#[1]

#ADD SIGMA YOU MUPPET
sigma = 15
vmin = np.median(master_bias)-3*sigma
vmax = np.median(master_bias)+3*sigma
#PLOTTING MASTERS
plt.subplot(311)
plt.imshow(master_bias, vmin = vmin, vmax = vmax) #masterBIAS
plt.subplot(312)
plt.imshow(master_dark, vmin = vmin, vmax = vmax) #masterDARK
plt.subplot(313)
plt.imshow(master_bias, vmin = vmin,vmax = vmax) #masterFLAT
plt.show()
