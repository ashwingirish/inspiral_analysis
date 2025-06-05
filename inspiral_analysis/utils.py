import numpy as np
from scipy.linalg import toeplitz
from astropy import constants as const
from pycbc.detector import Detector
import pycbc.waveform as pycbc_wf
import bilby
import lal

# Physical constants
G = const.G.value
M_SUN = const.M_sun.value
c = const.c.value


def time_domain_model(mass1, mass2, a_1, a_2, ld, theta_jn, phi_jl, tilt_1,
                      tilt_2, phi_12, phase, approximant):
    start_frequency = 20.0
    reference_frequency = 50.0

    m_1 = mass1 * M_SUN
    m_2 = mass2 * M_SUN

    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby.gw.conversion.bilby_to_lalsimulation_spins(
        theta_jn=theta_jn,
        phi_jl=phi_jl,
        tilt_1=tilt_1,
        tilt_2=tilt_2,
        phi_12=phi_12,
        a_1=a_1,
        a_2=a_2,
        mass_1=m_1,
        mass_2=m_2,
        reference_frequency=reference_frequency,
        phase=phase)

    delta_time = 1 / 4096.0
    hplus, hcross = pycbc_wf.get_td_waveform(approximant=approximant,
                                             mass1=mass1,
                                             mass2=mass2,
                                             spin1x=spin_1x,
                                             spin1y=spin_1y,
                                             spin1z=spin_1z,
                                             spin2x=spin_2x,
                                             spin2y=spin_2y,
                                             spin2z=spin_2z,
                                             distance=ld,
                                             inclination=iota,
                                             coa_phase=phase,
                                             delta_t=delta_time,
                                             f_lower=start_frequency,
                                             f_ref=reference_frequency)

    hp, hc = hplus.data, hcross.data
    tindex = np.argmax(np.sqrt(hp**2 + hc**2))
    hp = hp[0:tindex][-8192:]
    hc = hc[0:tindex][-8192:]

    if len(hp) < 8192:
        hp = np.pad(hp, (8192 - len(hp), 0), constant_values=hp[0])
        hc = np.pad(hc, (8192 - len(hc), 0), constant_values=hc[0])

    return hp, hc


def time_d(det_list, ref_det, ra, dec, tM_gps):
    td = {}
    ref_det_lal = lal.cached_detector_by_prefix[ref_det]
    for d in det_list:
        det = lal.cached_detector_by_prefix[d]
        td[ref_det + '_' + d] = lal.ArrivalTimeDiff(det.location,
                                                    ref_det_lal.location, ra,
                                                    dec, tM_gps)
    return td


def project(hp, hc, detector_name, ra, dec, psi, time):
    detector = Detector(detector_name)
    F_plus, F_cross = detector.antenna_pattern(ra, dec, psi, time)
    return F_plus * hp + F_cross * hc


def inner_product(a, b, inverse_covariance):
    return np.dot(a, np.dot(inverse_covariance, b))


class LogLikelihood:

    def __init__(self, event, det_class, time_delay, cov_inverse,
                 fixed_parametrs, priors, tgps, N, ref_det):
        self.event = event
        self.det_class = det_class
        self.time_delay = time_delay
        self.cov_inverse = cov_inverse
        self.fixed_parametrs = fixed_parametrs
        self.priors = priors
        self.tgps = tgps
        self.N = N
        self.ref_det = ref_det

        self.ra = fixed_parametrs[event]['ra']
        self.dec = fixed_parametrs[event]['dec']
        self.psi = fixed_parametrs[event]['psi']
        self.phi_jl = fixed_parametrs[event]['phi_jl']
        self.phi_12 = fixed_parametrs[event]['phi_12']
        self.phase = fixed_parametrs[event]['phase']
        self.trigger_time = tgps

    def __call__(self, theta):
        mass1, mass2, a_1, a_2, ld, theta_jn, tilt_1, tilt_2 = theta
        logL = 0.0

        for d in self.det_class.keys():
            dt = self.time_delay[f'{self.ref_det}_{d}']
            time_array_raw = self.det_class[d]['time'] - (self.trigger_time +
                                                          dt)
            data = self.det_class[d]['time_series'][time_array_raw <=
                                                    0.0][-self.N:]

            hp, hc = time_domain_model(mass1, mass2, a_1, a_2, ld, theta_jn,
                                       self.phi_jl, tilt_1, tilt_2,
                                       self.phi_12, self.phase,
                                       'IMRPhenomXPHM')

            prediction = project(hp, hc, d, self.ra, self.dec, self.psi,
                                 self.tgps)

            dd = inner_product(data, data, self.cov_inverse[d])
            dh = inner_product(data, prediction, self.cov_inverse[d])
            hh = inner_product(prediction, prediction, self.cov_inverse[d])
            logL += -0.5 * (dd - 2 * dh + hh)

        return logL


class PriorTransform:

    def __init__(self, priors, event):
        self.priors = priors
        self.event = event

    def __call__(self, utheta):
        m1_m, m2_m, a1_m, a2_m, ld_m, tj_m, t1_m, t2_m = utheta
        m1_min, m1_max = self.priors[self.event]['m1']
        m2_min = self.priors[self.event]['m2'][0]
        ld_min, ld_max = self.priors[self.event]['ld']

        mass1 = m1_min + (m1_max - m1_min) * m1_m
        mass2 = m2_min + (mass1 - m2_min) * m2_m
        a1 = 0.99 * a1_m
        a2 = 0.99 * a2_m
        ld = ld_min + (ld_max - ld_min) * ld_m
        theta_jn = np.pi * tj_m
        tilt_1 = np.pi * t1_m
        tilt_2 = np.pi * t2_m

        return mass1, mass2, a1, a2, ld, theta_jn, tilt_1, tilt_2
