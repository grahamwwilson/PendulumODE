# PendulumODE

python Simulate.py

Leads to files called 

1) Oscillations.dat

This consists of a table with the ODE solution for 1000 steps 
per oscillation 

2) SimDataFile.dat

Simulated transition information in similar format to pendulum timer 
data.

Examples with default parameters are:

SimDataFile-0.dat.gz
and
Oscillations-0.dat.gz

Update. In the oscillation while loop check whether the 
oscillation will lead to all 8 measurements - otherwise terminate it.
This can be a good indicator of whether the UP laser angles are 
modeled well.

Examples with 
python Simulate.py -n 200 leading to 196 oscillations (the last 4 oscillations 
are not simulated because of the above check) are 

SimDataFile-101.dat.gz
and
Oscillations-101.dat.gz
