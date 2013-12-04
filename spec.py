from time import *
import pylab as plt
from numpy import *
from numpy import random
from scipy import integrate
from scipy.integrate import romberg
from scipy.interpolate import interp1d
from scipy.constants import c,h,k

start_time = time()

H_0 = 70.0
W_M = 0.3
W_A = 0.7
time_test = linspace(1,80,80)
g_zp = 20.72
r_zp = 21.49
i_zp = 22.16
z_zp = 22.62

###### Filter Area ################################

g_filter = loadtxt('g_filter_snls.dat', dtype=object)
r_filter = loadtxt('r_filter_snls.dat', dtype=object)
i_filter = loadtxt('i_filter_snls.dat', dtype=object)
z_filter = loadtxt('z_filter_snls.dat', dtype=object)

n_g = int(size(g_filter)/2.0)
n_r = int(size(r_filter)/2.0)
n_i = int(size(i_filter)/2.0)
n_z = int(size(z_filter)/2.0)
A_g = zeros(n_g)
A_r = zeros(n_r)
A_i = zeros(n_i)
A_z = zeros(n_z)
DRE_g = []
DRE_r = []
DRE_i = []
DRE_z = []
	
for i in range(0,n_g):
	x = g_filter[i][0]
	y = g_filter[i][1]
	A_g[i] = float(x)
	DRE_g.append(float(y))
	
for i in range(0,n_r):
	x = r_filter[i][0]
	y = r_filter[i][1]
	A_r[i] = float(x)
	DRE_r.append(float(y))
	
for i in range(0,n_i):
	x = i_filter[i][0]
	y = i_filter[i][1]
	A_i[i] = float(x)
	DRE_i.append(float(y))
	
for i in range(0,n_z):
	x = z_filter[i][0]
	y = z_filter[i][1]
	A_z[i] = float(x)
	DRE_z.append(float(y))

gfil_Area = trapz(DRE_g, A_g)
rfil_Area = trapz(DRE_r, A_r)
ifil_Area = trapz(DRE_i, A_i)
zfil_Area = trapz(DRE_z, A_z)
A_g *= 1e-10
A_r *= 1e-10
A_i *= 1e-10
A_z *= 1e-10

###################################################

R = lambda t: 2.7*(1.8e12) + 1.03*(1.8e12)*t
T = lambda t: 16270.0 - 186*t
	
def flux(t,z): # Planck Equation
	blueshift = A_i/(1.0+z)
	result = 2.0*h*(c**2)/(blueshift**5)
	result /= (exp(h*c/(blueshift*k*T(t))) - 1.0)
	return result

def lum(t,z): # function of time (days)
	return flux(t,z)*(R(t)**2)

def L_dist(z):
	func = lambda z: 1/((W_M*((1+z)**3)+W_A)**0.5)
	integral = integrate.romberg(func, 0.0, z)
	D_C = (c/H_0)*integral
	D_L = (1+z)*D_C
	return D_L/1000.0

###### Calculating the Luminosity at Earth ########
	
def L_earth(t,z):
	return lum(t,z)/((L_dist(z)*3.08567758e22)**2.0)

###### Redshifted #################################

def L_redshifted(t,z):
	return L_earth(t,z)/(1.0+z)
	
###### Multiplying SED by filter Response #########

def filter_spec(t,z):
	return L_redshifted(t,z)*DRE_i
	
###### Integration through filter #################

def spec_integral(t,z):
	return trapz(filter_spec(t,z), A_i)
	
###### Magnitude ##################################

def mag(t,z):
	return -2.5*log10((spec_integral(t,z)*1e3)/ifil_Area) - i_zp
	
##### Fake Supernovae #############################

def SN(z):
	time = arange(0,80,(1/(1+z)))
	SuperNovae = zeros(len(time))
	x = 0.0
	for i in time:
		SuperNovae[x] = mag(i,z)
		x = x + 1.0
	return SuperNovae, time
	
##### Random dates/redshifts ######################

def monte_carlo(n):
	field, epoch, julian_date = loadtxt('obslog.txt', dtype=object, unpack=True)
	mag_array = zeros((len(epoch),73))
	probab_array = zeros((len(epoch),73))
	for i in range(0,len(epoch)):
		if field[i] == 'D1':
			specific_mag, specific_probab = loadtxt('/Users/thomasedwards/Documents/MPhys_Project/Computing/attachment/'+'D1_'+epoch[i]+'.dat', dtype=float, unpack=True)
			for j in range(0, len(specific_mag)):
				mag_array[i][j] = specific_mag[j]
				probab_array[i][j] = specific_probab[j]
	redshift = 0.5
	s = 20
	efficiency = zeros(s)
	found1 = zeros(s)
	test_redshifts = zeros(s)
	for i in range(0,s):
		lightcurve, time = SN(redshift) 
		for j in range(0,n):
			detection_mag = []
			index = []
			start_date = round(random.uniform(52801.53, 53026.30),2)#52941.53
			dates = time*(1+redshift) + start_date
			for k in range(0,len(epoch)):
				for m in range(0,len(dates)):
					if field[k] == 'D1':
						#print int(float(julian_date[k])), int(dates[m])
						if int(dates[m]) == int(float(julian_date[k])):
						#	print int(dates[m])
						#	if detection_limit(mag_array[k,:],probab_array[k,:],lightcurve[m]) == True:
							detection_mag.append(lightcurve[m])
							index.append(k)
			#print len(detection_mag), 'before'
			if found(mag_array, probab_array, index, detection_mag) == True:
				#print detection_mag, 'after'
				found1[i] = found1[i] + 1.0
				#print detection_mag
				detection_new = detection_limit(mag_array, probab_array, index, detection_mag)
				#print detection_new
				if useful(detection_new) == True:
						efficiency[i] = efficiency[i] + 1.0
		found1[i] = found1[i]/n					
		efficiency[i] = efficiency[i]/n
		test_redshifts[i] = redshift
		print redshift, found1[i], efficiency[i]
		redshift = redshift + 0.1
	return efficiency, found1, test_redshifts
		
##### Detection Criterion #########################

def useful(detection_mag):
	if len(detection_mag) == 0:
		return False
	length = len(detection_mag)
	min_point = detection_mag.index(min(detection_mag))
	if length >= 5 and min_point > 1 and min_point < (length - 1):
		return True
	else:
		return False
	
##### Probability of being found ##################

def found(mag_array, probab_array, index, magnitude):
	x = 0.0
	#print 'new'
	for i in range(0,len(index)):
		prob_limit = interp1d(mag_array[index[i],:], probab_array[index[i],:])
		#print prob_limit(magnitude[i])
		chance = random.random()
		#print magnitude[i], chance#, prob_limit(magnitude[i])
		if magnitude[i] > 25.75:
			continue
		if chance <= prob_limit(magnitude[i]):
			return True
	if x == 0.0:
		return False
		
##### Detection Limit #############################
	
def detection_limit(mag_array, probab_array, index, magnitude):
	for j in range(0,len(index)):
		probab_array_new = probab_array[index[j],:]
		mag_array_new = mag_array[index[j],:]
		reverse_probab = probab_array_new[::-1]
		reverse_mag = mag_array_new[::-1]
		for i in range(0,len(probab_array)):
			if reverse_probab[i] >= 0.5:
				x = i
				break
			else:
				x = 0.0
		if x == 0.0:
			magnitude[j] = 0.0
		interpol_mag = zeros(2)
		interpol_probab = zeros(2)
		place = 0.0
		for i in range(x-1,x+1):
			interpol_mag[place] = reverse_mag[i]
			interpol_probab[place] = reverse_probab[i]
			place = place + 1.0	
		prob_limit = interp1d(interpol_probab, interpol_mag)
		mag_limit = prob_limit(0.5)
		#print mag_limit
		if magnitude[j] >= mag_limit:
			magnitude[j] = 0.0
		else:
			continue
	magnitude[:] = (value for value in magnitude if value != 0.0)
	return magnitude

X, Y, Z = monte_carlo(10000)
print time() - start_time, "seconds"

plt.plot(Z,X)
#plt.plot(Z,Y)
plt.xlabel('Redshift')
plt.ylabel('Efficiency')
#plt.title('Efficiency Grpah D2 - Season 1 - 10000')
#plt.gca().invert_yaxis()
plt.show() 