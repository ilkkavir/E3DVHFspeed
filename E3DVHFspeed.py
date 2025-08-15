#
# A simple script to model measurement errors in bistatic measurements
# with EISCAT VHF and the EISCAT3D remote sites with different beam widths. 
#
#
# IV 2024
#

import e3doubt
import numpy as np
import datetime
from e3doubt.utils import get_supported_sites
import matplotlib.pyplot as plt
plt.ion()

##################
## Modify these ##
##################

# Remote receiver beam widths to test with
bwidths = (2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.,6.5,7,7.5,8)

# reference data from IRI model
refdate_models = datetime.datetime(2020,6,20,11,0,0)

# Altitudes
hh = 300.

# final integrated range resolution at each alttiude
resR = 12.

# range resolution from modulation (bit length) in km (3 km is matched with beata)
bitLength = 3.

# time resolution (the whole scan)
resT = 60.


#################################################################
## The settings below should not changes, but modify if needed ##
#################################################################

# beam width of the VHF radar
bwVHF = 1.5

# Site locations. I may be doing something unnecessary here, but this works. 
sites = get_supported_sites()

gdlat_t, gdlon_t = sites.loc['TRO']
gdlat_r1, gdlon_r1 = sites.loc['TRO']
gdlat_r2, gdlon_r2 = sites.loc['KAI']
gdlat_r3, gdlon_r3 = sites.loc['KRS']

TXloc = ('TRO',gdlat_t,gdlon_t)
RXloc = [('TRO',gdlat_r1,gdlon_r1),('KAI',gdlat_r2,gdlon_r2),('KRS',gdlat_r3,gdlon_r3)]

# radar carrier frequency
radarFreq  = 224e6

# transmitter duty cycle
dutyCycle = 0.125

# "reception" duty cycle
rxduty = [1.,1.,1.]

# system noise temperature.
Tnoise = 300.

# transmitter power
PTX = 1e6

# elevation limits for transmission and reception
mineleTrans = 30.
mineleRec = [30., 30., 30.]

# are the antennas phased-arrays? (beams widen when steered away from zenith)
phArrTrans = False
phArrRec = (False,True,True)

# transmitter beam width (full width at half maximum)
beamwidthTX = bwVHF


# transmitter beam azimuth array
TXaz = np.array([0])

# transmitter beam elevation array
TXel = np.array([90])

# a "layer thickness" to make approximate self-noise calcultions. 50 km has worked well.
NeThickness = 50.

# radar system parameters
RADAR_PARMS = dict(fradar=radarFreq,
                   dutyCycle=dutyCycle,
                   RXduty=rxduty,
                   Tnoise=Tnoise,
                   Pt=PTX,
                   mineleTrans=mineleTrans,
                   mineleRec=mineleRec,
                   phArrTrans=phArrTrans,
                   phArrRec=phArrRec,
                   fwhmRange=bitLength
                   )


dne1 = np.empty(len(bwidths))
dne2 = np.empty(len(bwidths))
dne3 = np.empty(len(bwidths))
dnemulti = np.empty(len(bwidths))
dTe1 = np.empty(len(bwidths))
dTe2 = np.empty(len(bwidths))
dTe3 = np.empty(len(bwidths))
dTemulti = np.empty(len(bwidths))
dTi1 = np.empty(len(bwidths))
dTi2 = np.empty(len(bwidths))
dTi3 = np.empty(len(bwidths))
dTimulti = np.empty(len(bwidths))
dVi1 = np.empty(len(bwidths))
dVi2 = np.empty(len(bwidths))
dVi3 = np.empty(len(bwidths))
dVimulti = np.empty(len(bwidths))


## test with all listed beam widths
ii = 0
for bw in bwidths:
    
    # receiver beam widths. 
    beamwidthRX = [bwVHF, bw , bw ]
    
    # initialize the radar experiments
    experiment = e3doubt.Experiment(az=TXaz,el=TXel,h=hh,refdate_models=refdate_models,transmitter=TXloc,receivers=RXloc,fwhmtx=beamwidthTX,fwhmrx=beamwidthRX,radarparms=RADAR_PARMS,resR=resR)
    
    
    # run IRI and MSIS
    experiment.run_models()
    
    # Calculate the parameter error estimates
    parerrs = experiment.get_uncertainties(integrationsec=resT,fwhmIonSlab=NeThickness)
    
    # coordinates of the measurement volumes
    points = experiment.get_points()

    # plasma parameter errors
    dne1[ii] = parerrs.loc[0,'dne1']
    dne2[ii] = parerrs.loc[0,'dne2']
    dne3[ii] = parerrs.loc[0,'dne3']
    dnemulti[ii] = parerrs.loc[0,'dnemulti']
    dTe1[ii] = parerrs.loc[0,'dTe1']
    dTe2[ii] = parerrs.loc[0,'dTe2']
    dTe3[ii] = parerrs.loc[0,'dTe3']
    dTemulti[ii] = parerrs.loc[0,'dTemulti']
    dTi1[ii] = parerrs.loc[0,'dTi1']
    dTi2[ii] = parerrs.loc[0,'dTi2']
    dTi3[ii] = parerrs.loc[0,'dTi3']
    dTimulti[ii] = parerrs.loc[0,'dTimulti']
    dVi1[ii] = parerrs.loc[0,'dVi1']
    dVi2[ii] = parerrs.loc[0,'dVi2']
    dVi3[ii] = parerrs.loc[0,'dVi3']
    dVimulti[ii] = parerrs.loc[0,'dVimulti']


    
    ii += 1
    


# plot the results
fig, axs = plt.subplots(2,2)
axs[0,0].plot(bwidths,dne1,'k-',label='VHF')
axs[0,0].plot(bwidths,dne2,'r-',label='KAI')
axs[0,0].plot(bwidths,dne3,'g-',label='KAR')
axs[0,0].plot(bwidths,dnemulti,'b-',label='Multistatic')
axs[0,0].set(ylabel=r'$\Delta N_e [m^{-3}]$')
axs[0,0].legend(frameon=False)

axs[0,1].plot(bwidths,dTe1,'k-',label='VHF')
axs[0,1].plot(bwidths,dTe2,'r-',label='KAI')
axs[0,1].plot(bwidths,dTe3,'g-',label='KAR')
axs[0,1].plot(bwidths,dTemulti,'b-',label='Multistatic')
axs[0,1].set(ylabel=r'$\Delta T_e [K]$')
axs[0,1].yaxis.tick_right()
axs[0,1].yaxis.set_label_position('right')

axs[1,0].plot(bwidths,dTi1,'k-',label='VHF')
axs[1,0].plot(bwidths,dTi2,'r-',label='KAI')
axs[1,0].plot(bwidths,dTi3,'g-',label='KAR')
axs[1,0].plot(bwidths,dTimulti,'b-',label='Multistatic')
axs[1,0].set(xlabel='Remote site beam width [deg]',ylabel=r'$\Delta T_i [K]$')

axs[1,1].plot(bwidths,dVi1,'k-',label='VHF')
axs[1,1].plot(bwidths,dVi2,'r-',label='KAI')
axs[1,1].plot(bwidths,dVi3,'g-',label='KAR')
axs[1,1].set(xlabel='Remote site beam width [deg]',ylabel=r'$\Delta V_i [ms^{-1}]$')

axs[1,1].yaxis.tick_right()
axs[1,1].yaxis.set_label_position('right')

fig.suptitle('Bit length: '+str(bitLength)+' km, Resol. : '+str(resR)+' km, Alt: '+str(hh)+' km\n IRI '+refdate_models.strftime("%Y/%m/%d %H:%M UTC"))

# save as png
fname = 'E3DVHFspeed_'+refdate_models.strftime("%Y%m%dT%H%M")+'_'+str(hh)+'km_'+str(bitLength)+'km_'+str(resR)+'km.png'
plt.savefig(fname,dpi=300)


