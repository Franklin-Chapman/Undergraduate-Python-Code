# Undergraduate-Python/MATLAB-Code

This repository consists of relevant Python programs that Franklin Chapman made during his undergraduate course work.  An explanation of the functionality of purpose of each program in this repository is to follow in this file.  

1. Franck Hertz large argon multi Gauss fit for loop python code.py
This program was used to analyze data for a lab performing the Franck Hertz experiment to find the first excitation energy of Argon gas.  The program imports a series of data files with the voltage and charge measured on the apparatus used in the experiment, it then fits the imported data using a function called multi_gauss defined at the top of the code.  After fitting the data it then saves the parameters of the fir function for a given data file and then these parameters from each data file are grouped together.  Finally with these groups the “mu” parameter (peak of each Gaussian curve) is used for the final part of the data analysis.

2. RBO reductions unfinished python code
This program was part of a research project to automate the first part of data analysis for the Red Buttes Observatory (RBO) which conducts followup observations of exoplanets for the University of Wyoming and Pennsylvania State University.  The program is unfinished, however does ⅓ of what it was intended to do.  The program navigates to a directory on the Linux computer systems at the University of Wyoming's Physics and Astronomy department where a given data set is stored.  It then opens the fit files of the exposures for the calibration images.  The single part that works is the creation of the Master Dark frame that takes roughly 5 dark frames, makes a 3D array, and averages over every pixel to make a master dark frame.  This was then to be done in similar ways for the flat and bias calibration images.  These then were to be subtracted off of the final science images.

3. cs-137 single gauss fit for loop python code
This program was used to analyze data for a lab studying the spectrums of a CS-137 radioactive sample.  The program works in a very similar way to the Franck Hertz data analysis program mentioned above.  It first imports a number of files (roughly 70 in total) with the number of counts per channel of a multichannel analyser measuring the signal from the radioactive sample.  It then takes this data and fits a single Gaussian function to the data and interively plots the data and fit function.  It then saves some select parameters of the fit function to then do the final step of the data analysis.  

4. dat file read and table generation python code 
This program was used for homework in an Astrophysics course to find the flux values from several star spectrums.

5. gamma exp co-60 double_peak_and_noise_fit python code
This program was used in the same experiment as with the CS-137 but with a CO-60 radioactive sample instead.  This sample spectrum was fitted using least squares curve fitting as it had two Gaussian peaks instead of the single peak that the CS-137 had.  This program also does the data analysis without the use of a for loop and this is responsible for the longer length of the program code.

6. pix brightness base python code
This program imports a photo and finds the brightness of a single pixel.

7. torsion balance fit method 1 python code
This program fits decaying harmonic data from a torsion balance used to re-create Cavendish's experiment to measure the gravitational constant.  After fitting in the same method as other programs above it then computes the gravitational constant.

8. torsion balance fit method 2 python code
A slightly different version of the above program used for the same experiment. 

