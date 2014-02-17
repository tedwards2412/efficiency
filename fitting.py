import numpy as np
import pylab as plt
from scipy.optimize import leastsq
from scipy import integrate
from scipy.integrate import romberg
from scipy.interpolate import interp1d
from scipy.stats import f
from scipy.constants import c,h,k,pi

H_0 = 70.0
W_M = 0.3
W_A = 0.7
z = 1.588
Mjd, Date, useless, Filter, Flux, Flux_error, Zeropoint = np.loadtxt('06D4eu_lc_ave.dat', dtype=object, unpack=True)

def data_reduction(Mjd,Filter,Flux,Errors):
	Mjd = [float(i) for i in Mjd]
	X = []
	Y_temp = [float(i) for i in Flux]
	Y = []
	error = []

	max_flux = np.max(Y_temp)
	max_index = Y_temp.index(max_flux)

	for i in range(0, len(Mjd)):
	#	if Mjd[i] < Mjd[max_index] + (250*(1+z)):
		X.append(Mjd[i] - np.min(Mjd))
		Y.append(float(Flux[i]))
		error.append(float(Flux_error[i]))

	Area = []
	DRE = np.zeros((514,len(Mjd)))
	A = np.zeros((514,len(Mjd)))

	for i in range(0, len(Mjd)):
		A_temp, DRE_temp = np.loadtxt(str(Filter[i])+'_filter_snls.dat',unpack=True)
		Area.append(np.trapz(DRE_temp, A_temp))
		for j in range(0, len(A_temp)):
			A[j,i] = A_temp[j]
			DRE[j,i] = DRE_temp[j]
	A *= 1e-10

	return X,Y,error, A, Area, DRE, np.min(Mjd)

############ Model 1 ##########################

def R(R0,dR,t):
	return R0 + dR*(1.0e12)*t

def T(T_0,dT,t):
	return T_0 - dT*t
	
def flux(T_0,dT,t,A): # Planck Equation
	blueshift = A/(1.0+z)
	result = (2.0*pi*h*(c**2))/(blueshift**5)
	result /= (np.exp(h*c/(blueshift*k*T(T_0,dT,t))) - 1.0)
	return result

def lum(R0,dR,T_0,dT,t,A): # function of time (days)
	return flux(T_0,dT,t,A)*(R(R0,dR,t)**2)

def L_dist(z):
	func = lambda z: 1/((W_M*((1+z)**3)+W_A)**0.5)
	integral = integrate.romberg(func, 0.0, z)
	D_C = (c/H_0)*integral
	D_L = (1+z)*D_C
	return D_L/1000.0
	
def L_earth(R0,dR,T_0,dT,t,A):
	return lum(R0,dR,T_0,dT,t,A)/((L_dist(z)*3.08567758e22)**2.0)

def L_redshifted(R0,dR,T_0,dT,t,A):
	return L_earth(R0,dR,T_0,dT,t,A)/(1.0+z)
	
def filter_spec(R0,dR,T_0,dT,t,A,DRE):
	return L_redshifted(R0,dR,T_0,dT,t,A)*(DRE)
	
def spec_integral(R0,dR,T_0,dT,t,A,DRE):
	Anew = []
	DREnew = []
	for i in range(0,len(A)):
		if A[i] > 0.0:
			Anew.append(float(A[i]))
			DREnew.append(float(DRE[i]))
	Anew = np.array(Anew)
	DREnew = np.array(DREnew)

	return np.trapz(filter_spec(R0,dR,T_0,dT,t,Anew,DREnew), Anew)*1e3

############ Model 2 ##########################

def R1(R0,dR,d2R,t):
	return R0 + dR*(1.0e12)*t + d2R*(1.0e12)*(t**2.0)

def T1(T_0,dT,t):
	return T_0 - dT*t
	
def flux1(T_0,dT,t,A): # Planck Equation
	blueshift = A/(1.0+z)
	result = 2.0*h*(c**2)/(blueshift**5)
	result /= (np.exp(h*c/(blueshift*k*T1(T_0,dT,t))) - 1.0)
	return result

def lum1(R0,dR,d2R,T_0,dT,t,A): # function of time (days)
	return flux1(T_0,dT,t,A)*(R1(R0,dR,d2R,t)**2)

def L_dist(z):
	func = lambda z: 1/((W_M*((1+z)**3)+W_A)**0.5)
	integral = integrate.romberg(func, 0.0, z)
	D_C = (c/H_0)*integral
	D_L = (1+z)*D_C
	return D_L/1000.0
	
def L_earth1(R0,dR,d2R,T_0,dT,t,A):
	return lum1(R0,dR,d2R,T_0,dT,t,A)/((L_dist(z)*3.08567758e22)**2.0)

def L_redshifted1(R0,dR,d2R,T_0,dT,t,A):
	return L_earth1(R0,dR,d2R,T_0,dT,t,A)/(1.0+z)
	
def filter_spec1(R0,dR,d2R,T_0,dT,t,A,DRE):
	return L_redshifted1(R0,dR,d2R,T_0,dT,t,A)*DRE
	
def spec_integral1(R0,dR,d2R,T_0,dT,t,A,DRE):
	Anew = []
	DREnew = []
	for i in range(0,len(A)):
		if A[i] > 0.0:
			Anew.append(float(A[i]))
			DREnew.append(float(DRE[i]))
	Anew = np.array(Anew)
	DREnew = np.array(DREnew)

	return np.trapz(filter_spec1(R0,dR,d2R,T_0,dT,t,Anew,DREnew), Anew)*1e3

#######################################################

X, Y, error, A, Area, DRE, firstobs_Mjd = data_reduction(Mjd,Filter,Flux,Flux_error)

############ Residuls1 ##########################

def residuals1(p,Y,X,error,A,Area,DRE): #date is an array
	array = []
	R0,dR,T_0,dT,t = p
	for i in range(0,len(X)):
		if X[i] < t:
			model = 0.0
		if X[i] > t:
			model = (spec_integral(R0,dR,T_0,dT,((X[i]-t)/(1+z)),A[:,i],DRE[:,i]))/Area[i]
		err = ((Y[i] - model)/(error[i]))
		array.append(err)
	#print np.sum(array)
	return array

p0 = [0.0, 2.0, 20000.0, 200.0, 5.0]
print 'Model 1'
res1 = leastsq(residuals1, p0, args=(Y,X,error,A,Area,DRE))
print 'R0 =', res1[0][0], 'dR =', res1[0][1], 'T0 =', res1[0][2], 'dT =', res1[0][3], 'Explosion day =', res1[0][4] + firstobs_Mjd 

############ Residuls 2 ##########################

def residuals2(p,Y,X,error,A,Area,DRE): #date is an array
	array = []
	R0,dR,d2R,T_0,dT = p
	for i in range(0,len(X)):
		if X[i] < 47.5216:
			model = 0.0
		if X[i] > 47.5216:
			model = (spec_integral1(R0,dR,d2R,T_0,dT,((X[i]-47.5216)/(1+z)),A[:,i],DRE[:,i]))/Area[i]
		err = ((Y[i] - model)/(error[i]))
		array.append(err)
	#print np.sum(array)
	return array

p1 = [0.0, 2.0, -2.0, 20000.0, 200.0]
print 'Model 2'
res2 = leastsq(residuals2, p1, args=(Y,X,error,A,Area,DRE))
print 'R0 =', res2[0][0], 'dR =', res2[0][1], 'd2R =', res2[0][2], 'T0 =', res2[0][3], 'dT =', res2[0][4]#, 'Explosion day =', res2[0][5] + firstobs_Mjd 

############################## Model Plotting
#
time = []
time1 = []
model = []
model1 = []
data = []
yflux = []
fluxerror = []
for i in np.arange(0,60,(1/(1+z))):
		time.append(i*(1+z)+res1[0][4])
		time1.append(i*(1+z)+47.5216)
		model.append((spec_integral(res1[0][0],res1[0][1],res1[0][2],res1[0][3],i,A[:,1],DRE[:,1]))/Area[1])
		model1.append((spec_integral1(res2[0][0],res2[0][1],res2[0][2],res2[0][3],res2[0][4],i,A[:,1],DRE[:,1]))/Area[1])
for i in range(0,15):
	data.append(X[i])
	yflux.append(Y[i])
	fluxerror.append(error[i])


################################## Chi-Square

def chi_square1(Y,X,error,A,Area,DRE,res):
	temp = np.zeros(len(X))
	for i in range(0, len(X)):
		if X[i] < res[0][4]:
			model = 0.0
		if X[i] > res[0][4]:
			model = (spec_integral(res[0][0],res[0][1],res[0][2],res[0][3],((X[i]-res[0][4])/(1+z)),A[:,i],DRE[:,i]))/Area[i]
		temp[i] = ((Y[i]-model)/error[i])**2.0
	return np.sum(temp)

def chi_square2(Y,X,error,A,Area,DRE,res):
	temp = np.zeros(len(X))
	for i in range(0, len(X)):
		if X[i] < 47.5216:
			model = 0.0
		if X[i] > 47.5216:
			model = (spec_integral1(res[0][0],res[0][1], res[0][2],res[0][3],res[0][4],((X[i]-47.5216)/(1+z)),A[:,i],DRE[:,i]))/Area[i]
		temp[i] = ((Y[i]-model)/error[i])**2.0
	return np.sum(temp)

Chi1 = chi_square1(Y,X,error,A,Area,DRE,res1)
Chi2 = chi_square2(Y,X,error,A,Area,DRE,res2)
df1 = len(X) - 5; df2 = len(X) - 6
F = ((Chi1-Chi2)/(df1-df2)) / (Chi2/df2)
p_value = f.cdf(F, df1, df2)

print 'Chi1 =', Chi1
print 'Chi2 =', Chi2
print 'df1 =', df1, 'df2 =', df2
print 'F statistic =', F
print 'p-value = ', p_value

if p_value > 0.95:
	print 'Accept Model 2'
if p_value < 0.95:
	print 'Accept Model 1'

#################################	

plt.plot(time, model, color='b')
plt.plot(time1, model1, color='r')
plt.errorbar(data, yflux, yerr=fluxerror, fmt='o', color='g')
#plt.gca().invert_yaxis()
plt.show()