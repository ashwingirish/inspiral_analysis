import sys
sys.path.append('/mnt/pfs/akash.mishra/paper/scripts/')
import matplotlib.pyplot as plt

from os import environ
import numpy as np, sys
from scipy.linalg import toeplitz

import bilby
import pycbc
import pycbc.waveform as pycbc_wf
# import waveform
import lal
import pickle
from astropy import constants as const
from multiprocessing import Pool
import dynesty
import dill
import numba
from numba import jit
import numpy as np
from pycbc.detector import Detector

# Constants
G = const.G.value
M_SUN = const.M_sun.value
c = const.c.value

############################
######## Functions #########
############################

def time_domain_model(mass1, mass2, a_1, a_2, ld, theta_jn, phi_jl, tilt_1, tilt_2, phi_12, phase, approximant):
    start_frequency = 20.0
    
    reference_frequency = 50 # waveform_kwargs.get('reference_frequency', 50.0)
    m_1 = mass1 * 1.9884099021470415e+30 # solar mass
    m_2 = mass2 * 1.9884099021470415e+30 # solar mass
    
        
    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby.gw.conversion.bilby_to_lalsimulation_spins(
        theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=m_1, mass_2=m_2, reference_frequency=reference_frequency, phase=phase)
        
    longitude_ascending_nodes = 0.0
    mean_per_ano = 0.0
    delta_time = 1/4096.0
        
    hplus, hcross = pycbc_wf.get_td_waveform( approximant = approximant,#'SpinTaylorT4',
            mass1=mass1, mass2=mass2, spin1x=spin_1x,  spin1y=spin_1y, spin1z=spin_1z, 
            spin2x=spin_2x, spin2y=spin_2y, spin2z=spin_2z, distance=ld, 
            inclination=iota, coa_phase=phase,  delta_t = delta_time,
            f_lower = start_frequency, f_ref=reference_frequency)
    
    hp = hplus.data
    hc = hcross.data
    tindex = np.argmax(np.sqrt(hp**2 + hc**2))
    hp = hp[0:tindex][-8192:]
    hc = hc[0:tindex][-8192:]
    
    if len(hp)<8192:
        prefix_hp = np.full(8192 - len(hp), hp[0])
        prefix_hc = np.full(8192 - len(hc), hc[0])
        hp = np.concatenate((prefix_hp, hp))
        hc = np.concatenate((prefix_hc, hc))
     
    
    # end_value = (len(hp) - 1) * (1/4096)
    # times = np.linspace(-end_value, 0.0, len(hp))
    # times = np.linspace(-2.0, 0.0, len(hp))

    return hp, hc


def mass_to_time(M):
    return (G*M*M_SUN)/(c**3)

# Time delay calculation function
def time_d(det_list, ref_det, ra, dec, tM_gps):
    td = {}
    ref_det_lal = lal.cached_detector_by_prefix[ref_det]
    for d in det_list:
        det = lal.cached_detector_by_prefix[d]
        td[ref_det + '_' + d] = lal.ArrivalTimeDiff(det.location, ref_det_lal.location, ra, dec, tM_gps)
    return td


def project(hp, hc, detector_name, ra, dec, psi, time):
    detector = Detector(detector_name)
    F_plus, F_cross = detector.antenna_pattern(ra, dec, psi, time)
    waveform = F_plus*hp + F_cross*hc

    return waveform

def inner_product(a, b, inverse_covariance):
    return np.dot(a, np.dot(inverse_covariance, b))

import pickle
with open(f'/mnt/pfs/akash.mishra/paper/data_files/events_and_detectors.pkl', 'rb') as file:
    event_details = pickle.load(file)
    
with open(f'/mnt/pfs/akash.mishra/paper/data_files/fixed_parameters.pkl', 'rb') as file:
    fixed_parametrs = pickle.load(file)
    
    
with open(f'/mnt/pfs/akash.mishra/paper/data_files/peak_time.pkl', 'rb') as file:
    peak_time = pickle.load(file)
    
with open(f'/mnt/pfs/akash.mishra/paper/data_files/priors.pkl', 'rb') as file:
    priors = pickle.load(file)
    
with open(f'/mnt/pfs/akash.mishra/paper/data_files/gw_data.pkl', 'rb') as file:
    gw_data = pickle.load(file)
    
with open(f'/mnt/pfs/akash.mishra/paper/data_files/final_mass_and_snr.pkl', 'rb') as file:
    final_mass_snr = pickle.load(file)
    
# List of events to process
events = ['GW150914', 'GW170729', 'GW190602_175927', 
          'GW200311_115853', 'GW200224_222234']

# Function to process a single event
def process_event(event):
    print(f"Processing event: {event}")
    
    
    N = 2*4096

    detectors = event_details[event]

    det_class = gw_data[event]
    cov_inverse = {}
   
    for i in detectors:
        cov_inverse[i] = np.linalg.inv(toeplitz(det_class[i].acf[0:N]))
      
    ra = fixed_parametrs[event]['ra']
    dec = fixed_parametrs[event]['dec']
    psi = fixed_parametrs[event]['psi']
    
    #theta_jn = 2.69
    phi_jl = fixed_parametrs[event]['phi_jl']
    # tilt_1 = 1.68
    # tilt_2 = 1.77
    phi_12 = fixed_parametrs[event]['phi_12']
    phase = fixed_parametrs[event]['phase']
    trigger_time =  peak_time[event]
    tgps = trigger_time
    t_merger = 0.0
    
    ref_det = event_details[event][0]
    time_delay = time_d(event_details[event], ref_det, ra, dec, trigger_time)    # event_details[event] = ['H1', 'L1']

    # Log-likelihood function
    
    def loglikelihood(theta):
        mass1, mass2, a_1, a_2, ld, theta_jn, tilt_1, tilt_2  = theta
        #mass1, mass2, a_1, a_2, ld, theta_jn, phi_jl, tilt_1, tilt_2, phi_12, phase   = theta
    
        logL = 0.0

        for d in det_class.keys():
            #lal_det = lal.cached_detector_by_prefix[d]
            dt   = time_delay['{0}_'.format(ref_det)+d]
            tref = lal.LIGOTimeGPS(t_merger+dt+trigger_time)
            
            time_array_raw = det_class[d].time - (trigger_time+dt)
            time_array     = time_array_raw[time_array_raw <= t_merger][-N:]
            data           = det_class[d].time_series[time_array_raw <= t_merger][-N:]
            
            hp, hc =  time_domain_model(mass1, mass2, a_1, a_2, ld, theta_jn, phi_jl, tilt_1, tilt_2, phi_12, phase, 'IMRPhenomXPHM')
            prediction = project(hp, hc, d, ra, dec, psi, tgps)
            environ['OMP_NUM_THREADS'] = '8'
            dd = inner_product(data,       data,       cov_inverse[d])
            dh = inner_product(data,       prediction, cov_inverse[d])
            hh = inner_product(prediction, prediction, cov_inverse[d])
            residuals_inner_product = dd - 2. * dh + hh
            logL += -0.5*residuals_inner_product #+ log_normalisation
        return logL

    

    # prior transform
    def prior_transform(utheta):
        mass1_m, mass2_m, a_1m, a_2m, ld_m, theta_jn_m, tilt_1_m, tilt_2_m = utheta
        #mass1_m, mass2_m, a_1m, a_2m, ld_m, theta_jn_m, phi_jl_m, tilt_1_m, tilt_2_m, phi_12_m, phase_m = utheta
        
        m1_min = priors[event]['m1'][0]
        m1_max = priors[event]['m1'][1]
        
        m2_min = priors[event]['m2'][0]
        ld_min = priors[event]['ld'][0]
        ld_max = priors[event]['ld'][1]
        
        mass_1 = m1_min + (m1_max-m1_min) * mass1_m
    
    
    
        mass_2 = m2_min + (mass_1-m2_min) * mass2_m
    
        a_1 =  0.0 + (0.99 - 0.0) * a_1m
        a_2 =  0.0 + (0.99 - 0.0) * a_2m
        ld = ld_min + (ld_max-ld_min) * ld_m
        theta_jn = 0.0 + (np.pi - 0.0) * theta_jn_m
        #phi_jl = 0.0 + (2*np.pi - 0.0) * phi_jl_m
        tilt_1 = 0.0 + (np.pi - 0.0) * tilt_1_m
        tilt_2 = 0.0 + (np.pi - 0.0) * tilt_2_m
    
        #phi_12 = 0.0 + (2*np.pi - 0.0) * phi_12_m
        #phase = 0.0 + (2*np.pi - 0.0) * phase_m
    
        return mass_1, mass_2, a_1, a_2, ld, theta_jn, tilt_1, tilt_2

    # Dynesty sampler
    dsampler = dynesty.NestedSampler(loglikelihood, prior_transform, ndim=8, sample='unif', nlive=3000)
    dsampler.run_nested()

    dres = dsampler.results

    # Save the results for each event
    output_file = f'{event}_results.pkl'
    with open(output_file, 'wb') as f:
        dill.dump(dres, f)

    print(f"Completed event: {event}")
    
n_pool = len(events)

if __name__ == '__main__':
    with Pool(n_pool) as p:  # n_pool cores
        results = p.map(process_event, events)

    print("All events processed.")
