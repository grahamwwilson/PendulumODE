# Solve pendulum differential equation numerically.
# Here the ODE includes the exact sin(theta) term 
# (no small angle approximation) and linear and quadratic drag terms 
# following equation 3.1 of McInerney (see Expt 2 writeup). 
# Note that the quadratic term is written as omega*abs(omega) 
# in order that it opposes the motion on each half cycle.
#
# The initial code evaluated the solution for fixed time steps. 
# Now the code does half-period calculations.
# Needs some more work to calculate t values 
# corresponding to particular theta values and/or phases.
# This is now done for the simplest case.
#
# The simulation is simulating a similar 
# setup to the "Two-Timer Pendulum Measurements" writeup, (see Figure) 
# where one laser beam is centered on the equilibrium 
# position ("downstream" one), and the other is displaced "upstream".
# This leads to eight pendulum transition events per cycle, 
# four for each laser. In the simulated case, the bob is launched 
# from, theta = theta_0, and encounters the upstream laser centered 
# on theta = theta_up, and the downstream laser centered near theta = 0 mrad. 
import numpy as np
import math
import random
from myrandom import *
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import special
import argparse

# function only depends on w0sq, K, gamma. 
# But send longer list of parameters for compatibility with Reynolds number based approach of function f2.
def f(y, t, params):
    theta, omega = y          # unpack current values of y
    w0sq, K, gamma, L, rhoF, mu, D, mass, Irot = params   # unpack parameters
    derivs = [omega,          # list of dy/dt=f functions
             -w0sq*np.sin(theta) -K*omega*abs(omega) -gamma*omega]
    return derivs

# Currently only function f above is used in the code.
def f2(y, t, params):
# Use Cheng formula
    theta, omega = y            # unpack current values of y
    w0sq, K, gamma, L, rhoF, mu, D, mass, Irot = params   # unpack parameters
    u = abs(omega)*L            # flow speed
    Re = rhoF*u*D/mu            # Reynolds number
    A = (math.pi*D**2)/4.0      # sphere frontal area
    if Re > 0.003:
       Cd = (24.0/Re)*(1.0 + 0.27*Re)**0.43 + 0.47*(1.0-math.exp(-0.04*Re**0.38))
    else:
       Cd = 0.0
#    print('u, Re, Cd',u,Re,Cd)
    alphad = (0.5*Cd*rhoF*u*u*A)*L/Irot  # drag angular acceleration
    directiontest = omega*abs(omega)
    sign = 1.0
    if directiontest < 0.0:
       sign = -1.0
    derivs = [omega,            # list of dy/dt=f functions
             -w0sq*np.sin(theta) -alphad*sign]
    return derivs

parser = argparse.ArgumentParser(description='Pendulum Timer Simulator with ODE')
parser.add_argument("-n", "--noscs", type=int, default=130, help="Number of oscillations to simulate")
parser.add_argument("-d", "--drag", type=float, default=0.032, help="Drag")
parser.add_argument("-l", "--linearf", type=float, default=0.63, help="Linear fraction")
parser.add_argument("-q", "--theta0", type=float, default=0.54, help="Initial angle (rad)")
parser.add_argument("-t", "--ftup", type=float, default=0.2806, help="Up-stream laser angle fraction")
parser.add_argument("-e", "--effw", type=float, default=0.966, help="Up-stream laser effective width")
parser.add_argument("-s", "--steps", type=int, default=1000, help="ODE steps per oscillation")
parser.add_argument("-r", "--res", type=float, default=1.0e-6, help="Measurement time resolution")

args=parser.parse_args()
print('Found argument list: ')
print(args)
NOSCS = args.noscs      # Number of oscillations to simulate       
drag=args.drag          # Drag angular acceleration at equilibrium position [rad/s^2] 
linearf=args.linearf    # Linear drag fraction at equilibrium position
theta0=args.theta0      # Initial angle (rad)
FTHETAU=args.ftup       # Angle of up-stream laser as fraction of theta0
EFFWIDTHU=args.effw     # Effective width of up-stream laser
steps=args.steps        # ODE steps
TRMS =args.res          # Time resolution in seconds

# Toy MC parameters
TOFFSET = 0.0     # Set start-time of oscillation [seconds] 
# Clock frequency depends on which timer circuit is used
CLOCKFREQ1 = 5.013290e6 # Hz (calibrated approximately using computer clock-time 
                        #     between (D2,3) and (D1034,1035) for Run 76.) 
#CLOCKFREQ0 = CLOCKFREQ1/1.00057665 # Hz  (Measured using run 83 asymmetry)
CLOCKFREQ0 = CLOCKFREQ1/1.0005862 # Hz  (Measured using run 76 asymmetry)
CLOCKFREQU = CLOCKFREQ0 # Swapped configuration of run 76
CLOCKFREQD = CLOCKFREQ1 # Swapped configuation of run 76
SEED = 202
random.seed(SEED)   # Initialize random number
simdatafile = open("SimDataFile.dat", "w")
oscillationsfile = open("Oscillations.dat", "w")
print('# i       time[s]       theta[rad]      omega[rad/s]      ',
      ' Energy[%]       theta_0[rad]        omega_0[rad/s]       ',
      'drag-induced angular acceleration  [rad/s^2]',
      file=oscillationsfile)  # File header

# Parameters
G = 9.79971      # acceleration due to gravity [m/s^2]
# g calculated from https://www.ngs.noaa.gov/cgi-bin/grav_pdx.prl
# using (Latitude=38.97667N, Longitude=95.29868W, Elevation=880 ft). 
# Supposed uncertainty is 0.00002 m/s^2.
D = 0.0253       # bob diameter [m]
L = 0.33474      # pendulum length [m]
TFACTOR=0.9999634350018175 # Factor to adjust length to give correct average period
L=L*TFACTOR**2
    
omega0 = 0.0       # initial angular velocity [rad/s]
EoverM0 = G*L*(1.0 - np.cos(theta0))  # Initial value of E/m

R=D/2.0
IFACTOR = 1.0 + 0.4*(R/L)**2  # Include factor for increased rotational inertia 
                              # associated with extended body rather than point mass
                              # IFACTOR = 1.0005600 (for D=0.0253, L=0.338078)
w0sq = G/(L*IFACTOR)          # pendulum angular frequency squared [rad^2/s^2]
wmaxsq = 2.0*w0sq*(1.0-math.cos(theta0))  # max angular velocity squared assuming no drag
# Configure with given fractions of linear and quadratic drag at first equilibrium position
# (computed neglecting drag on 1/4 cycle from theta=theta_0 to theta=0 --- equivalent to 
#  starting the pendulum with initial conditions of theta=0, and same initial E).
quadraticf = 1.0 - linearf    # quadratic drag fraction at equilibrium position
gamma = drag*linearf/math.sqrt(wmaxsq)   # linear drag parameter
K = drag*quadraticf/wmaxsq               # quadratic drag parameter
print('Defined parameters ',G,L,theta0,wmaxsq,gamma,K)
# TODO Sphere drag model could perhaps be made more realistic using Cheng's 
# 5-parameter sphere drag coefficient parametrization.

# Derived quantities.
Tinf = 2.0*np.pi*np.sqrt(L/G)
w0 = np.sqrt(w0sq)
Q = 0.0
if gamma > 0.0:
   Q = w0/gamma
w1 = np.sqrt(w0sq - 0.25*gamma*gamma) # reduced frequency in presence of linear drag
k = np.sin(0.5*theta0)
T0 = (2.0*np.pi/w1)
Tguess = T0*(2.0/np.pi)*special.ellipk(k*k) # Estimate of the initial period
print('Derived quantities: w0sq, Tinf, w0, Q, w1, Tguess',w0sq,Tinf,w0,Q,w1,T0,Tguess)

rhoF = 1.204     # Air density (kg/m**3)
mu  = 1.825e-5   # Dynamic viscosity (kg/(m*s)) 

# Bob mass
rhoB = 8600.0    # Density of brass
mass = rhoB*(4.0/3.0)*math.pi*R**3
# reduce the mass a bit
freduce = 0.125
Irot = freduce*mass*(L**2 + 0.4*R**2)         # Rotational inertia

# Bundle parameters and initial conditions for ODE solver
params = [w0sq, K, gamma, L, rhoF, mu, D, mass, Irot]   # parameters
y0 = [theta0, omega0]       # initial conditions 

evolutionStep = 0
tStart = 0.0
tStop  = Tguess     # Do one full cycle at a time.
tStart_list = []
tStart_list.append(tStart)
tStop_list = []
tStop_list.append(tStop)
dt=999.0

TOLERANCE = 1.0e-12

TUP  = theta0*FTHETAU # Upstream laser position
# We know the upstream laser is not centered on the bob height.
# So effective width of the transition is smaller
TMAX = TUP + EFFWIDTHU*(0.50*D/L)   # upstream laser at TUP rad          
TMIN = TUP - EFFWIDTHU*(0.50*D/L)   #  "

TDOWN = -0.001            # small offset of downstream laser from equilibrium position
TARGET_THETAP = TDOWN+0.50*D/L
TARGET_THETAM = TDOWN-0.50*D/L

times_list = []
theta_list = []
omega_list = []
energy_list= []
# target values of theta for each cycle
targett   = [TMAX, TMIN, TARGET_THETAP, TARGET_THETAM, TARGET_THETAM, TARGET_THETAP, TMIN, TMAX]
# target sign of omega=dtheta/dt for corresponding target theta
targeto   = [ -1, -1, -1, -1, +1, +1, +1, +1 ]
# event type ("U" for upstream laser, "D" for downstream laser)
eventtype = [  "U",  "U",  "D",  "D",  "D", "D",  "U",  "U" ]

# Calculate the energy
EoverM = G*L*(1.0 - np.cos(TMAX) )
Ecritical = 1.0005*100.0*(EoverM/EoverM0)
Efnext =100.0  # Initialize
print('Ecritical set to ',Ecritical)

print(targett)
print(targeto)

# Search for t values that yield theta values as close as 
# possible to the requested value (targett) 
# with the requested sign of omega (targeto)

while evolutionStep < NOSCS and Efnext > Ecritical:  # controls number of complete cycles (8 measurements per cycle)
                                                     # Only start new cycle if new cycle is estimated to lead to each of the 8 measurements
   for x in range(len(targett)):
       targt = targett[x]
       targo = targeto[x]
       dt = 999.0
       print('x loop',targt,targo)

       ntries = 0

       while abs(dt) > TOLERANCE  and ntries < 100:
           ntries = ntries + 1
# Make time array for solution for this iteration from tStart to tStop
           numSteps = steps + 1
           t=np.linspace(tStart,tStop,numSteps)
# Call the ODE solver
           psoln = odeint(f, y0, t, rtol=0.5e-12, atol=0.5e-12, args=(params,))
# Find which step is closest to the required theta value
           dtheta = 999.0
           nrows,ncols = psoln.shape
           isel = -1
           for i in range(nrows):
               tvaluei = t[i]
               thetai = psoln.item(i,0)
               omegai = psoln.item(i,1)
               if abs(thetai - targt) < dtheta and omegai*targo > 0.0:
                  dtheta = abs(thetai - targt)
                  isel = i
# Record best value of i
           tvalue = t[isel]
           theta = psoln.item(isel,0)
           omega = psoln.item(isel,1)
           print('NR with t = ',tvalue,' theta = ',theta,' omega = ',omega)
# Evaluate correction needed for t
           dt = - (theta - targt)/(omega)
           print('NR dt = ',dt)
           tStopP = tvalue + dt
# Only make update if we will evolve the ODE at least one more time.
# (We need to re-initialize y0 when doing more cycles ... and we need 
#  to keep this consistent with our estimate of t)
           if abs(dt) > TOLERANCE:
              tStop = tStopP
           else:
# Tolerance is satisfied.
# Keep track of our best estimates of the period at the turning times 
# including the additional correction from time increment dt.
# Note that the theta value stored is not updated for 
# movement in time increment dt
              times_list.append(tStopP)
              theta_list.append(theta)
              omega_list.append(omega)
# Includes non-point mass in kinetic energy term...
              energy = G*L*(1.0 - np.cos(theta)  + 0.5*omega**2/w0sq )
              energy_list.append(energy)
              theta0estimate = math.acos(1.0 - (energy/(G*L)))
# Next loop - reset 
       print('ntries =',ntries)
       if ntries==100:
          print('Non-convergence ',evolutionStep,targt,targo,tStart,tStop)
       tStart = tStart_list[evolutionStep]
       tStop  = tStop_list[evolutionStep]
   evolutionStep +=1
# First advance by one cycle
   numSteps = steps + 1
   ilast = numSteps - 1
   t=np.linspace(tStart_list[evolutionStep-1],tStop_list[evolutionStep-1],numSteps)
# Call the ODE solver
   psoln = odeint(f, y0, t, rtol=0.5e-12, atol=0.5e-12, args=(params,))
# Update the initial values of the IVP to those from the latest iteration
   y0 = [psoln.item(ilast,0), psoln.item(ilast,1)]
# reset tStart, tStop
   tStart = tStop
# Recalculate Tguess based on updated amplitude estimate
   k = np.sin(0.5*theta0estimate)
   Tguess = T0*(2.0/np.pi)*special.ellipk(k*k)
   tStop = tStart + Tguess  # should correct this a bit...
   tStart_list.append(tStart)
   tStop_list.append(tStop)
# this loop stops when we have done the required number of oscillations

# But first, before the next iteration, we also save the information 
# from each iteration to a file
   Energy = np.empty(nrows, dtype=float)
   Theta = np.empty(nrows, dtype=float)
   ThetaMax = np.empty(nrows, dtype=float) # Keep track of energy equivalent amplitude
   OmegaMax = np.empty(nrows, dtype=float) # Keep track of energy equivalent maximum angular velocity 
   DragAngAcc = np.empty(nrows, dtype=float)
   for i in range(nrows):
      tvalue = t[i]
      theta = psoln.item(i,0)
      omega = psoln.item(i,1)
# Calculate the energy
      EoverM = G*L*(1.0 - np.cos(theta)  + 0.5*omega**2/w0sq )
      Efraction = 100.0*(EoverM/EoverM0)
      Energy[i] = Efraction
      Theta[i] = theta
      ThetaMax[i] = math.acos(1.0 - (EoverM/(G*L)))
      OmegaMax[i] = math.sqrt(2.0*w0sq*EoverM/(G*L))
      DragAngAcc[i] = -K*omega*abs(omega) - gamma*omega
      print('Time step ',i,'t= ',tvalue,' theta = ',theta,' omega = ',
             omega,'Energy fraction (%) = ',Efraction,' (q,w)_0 = ',ThetaMax[i],OmegaMax[i])
# Also save info to a file for later plotting
      if i==0 and evolutionStep==1:
         print(i,tvalue,theta,omega,Efraction,ThetaMax[i],OmegaMax[i],DragAngAcc[i],file=oscillationsfile)
      elif i!=0:
         print(steps*(evolutionStep-1)+i,tvalue,theta,omega,Efraction,ThetaMax[i],OmegaMax[i],DragAngAcc[i],file=oscillationsfile)
# Estimate the energy fractionafter the next complete oscillation
   Efnext = Energy[nrows-1]*(Energy[nrows-1]/Energy[0])
   print('Efnext = ',Efnext)
# We continue to loop until the number of oscillations requested are completed.

# Plot results. As it stands ALL these are just for the final oscillation
# 3. Add some plotting customization
SMALL_SIZE = 20
MEDIUM_SIZE = 26
BIGGER_SIZE = 32
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig = plt.figure(1, figsize=(8,8))

# Plot theta as a function of time
ax1 = fig.add_subplot(311)
ax1.plot(t, psoln[:,0])
ax1.set_xlabel('time')
ax1.set_ylabel('theta')

# Plot omega as a function of time
ax2 = fig.add_subplot(312)
ax2.plot(t, psoln[:,1])
ax2.set_xlabel('time')
ax2.set_ylabel('omega')

# Plot omega vs theta
ax3 = fig.add_subplot(313)
twopi = 2.0*np.pi
ax3.plot(psoln[:,0], psoln[:,1], '.', ms=1)
ax3.set_xlabel('theta')
ax3.set_ylabel('omega')
ax3.set_xlim(-1.05*theta0, 1.05*theta0)

plt.tight_layout()
#plt.show()

nrows, ncols = psoln.shape
print(nrows)
print(ncols)

plt.figure(4)
#plt.plot(t,Energy,'bo')
plt.plot(t,Energy,'bo',markersize=3)
plt.plot(t,Energy,linewidth=2,color='magenta')
plt.plot(t,Energy,'bo',markersize=3)
plt.title('Energy(%) vs time')
plt.xlabel('Time [s]')
plt.ylabel('Energy(%) relative to t=0')
#plt.show()

plt.figure(5)
plt.plot(t,ThetaMax,'bo')
plt.title('Energy equivalent amplitude (rad) vs time')
plt.xlabel('Time [s]')
plt.ylabel('Angular amplitude [rad]')
#plt.show()

plt.figure(6)
plt.plot(t,OmegaMax,'bo')
plt.title('Energy equivalent maximum angular velocity (rad/s) vs time')
plt.xlabel('Time [s]')
plt.ylabel('Maximum Angular Velocity [rad/s]')
#plt.show()

plt.figure(7)
plt.plot(Theta,DragAngAcc,'bo')
plt.title('Drag Angular Acceleration vs Theta')
plt.xlabel('Theta [rad]')
plt.ylabel('Drag Angular Acceleration [rad/s^2]')
#plt.show()

plt.figure(8)
plt.plot(t,DragAngAcc,'bo')
plt.title('Drag Angular Acceleration vs t')
plt.xlabel('t [s]')
plt.ylabel('Drag Angular Acceleration [rad/s^2]')
#plt.show()

# Add header line to output files (ignorable by python numpy etc).
print('# eventtype event clockticks time[s]  Energy[%]',file=simdatafile)

# Should add some checks of number of occurrences of non-monotonically increasing times
tprevious = -1.0
nproblems = 0
for x in range(len(times_list)):
    print('List item',x,times_list[x],theta_list[x],omega_list[x])
    tcurrent = times_list[x]
    if tcurrent < tprevious:
       print('Problem for tcurrent = ',tcurrent)
       nproblems +=1
    tprevious = tcurrent 
    if x%2==1:
       print('THETA',(x+1)//2, theta_list[x])
# Toy MC simulation
    tmeas = NormalVariate(tcurrent, TRMS)  # this function is in myrandom.py
    if eventtype[x%8]=="U":
       print('event type U')
       tmeasclockticks = int(tmeas*CLOCKFREQU + 0.5)  # Add 0.5 to avoid rounding bias 
    if eventtype[x%8]=="D":
       print('event type D')
       tmeasclockticks = int(tmeas*CLOCKFREQD + 0.5)  # Add 0.5 to avoid rounding bias
    efraction = 100.0*energy_list[x]/EoverM0
    print(eventtype[x%8],x,tmeasclockticks,tcurrent,efraction,file=simdatafile)

print('Number of problems = ',nproblems)

simdatafile.close()
oscillationsfile.close()

DTHETA = D/L
# Period calculations for first three oscillations
t2 = times_list[2]
t3 = times_list[3]
t4 = times_list[4]
t5 = times_list[5]
period0 = 4.0*( (t2+t3)/2.0 )
period1 = 2.0 * ( ((t4+t5)/2.0) - ((t2+t3)/2.0) )
print('Period 0 (  0 -  90) = ',period0,' ang. speed (rad/s) ',DTHETA/(t3-t2))
print('Period 1 ( 90 - 270) = ',period1,' ang. speed (rad/s) ',DTHETA/(t5-t4))
if NOSCS >=3:
   t10 = times_list[10]
   t11 = times_list[11]
   t12 = times_list[12]
   t13 = times_list[13]
   period2 = (t10+t11) - (t4+t5)
   period3 = (t12+t13) - (t10+t11)
   t18 = times_list[18]
   t19 = times_list[19]
   t20 = times_list[20]
   t21 = times_list[21]
   period4 = (t18+t19) - (t12+t13)
   period5 = (t20+t21) - (t18+t19)
   print('Period 2 (270 - 450) = ',period2, DTHETA/(t11-t10))
   print('Period 3 (450 - 630) = ',period3, DTHETA/(t13-t12))
   print('Period 4 (630 - 810) = ',period4, DTHETA/(t19-t18))
   print('Period 5 (810 - 990) = ',period5, DTHETA/(t21-t20))
   periods = 0.5*( (t10+t11) - (t2+t3) )
   print('Period S ( 90 - 450) = ',periods)

