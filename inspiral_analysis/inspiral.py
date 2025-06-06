import os
import sys, dill, pickle, dynesty
import numpy as np
from scipy.linalg import toeplitz
from utils import LogLikelihood, PriorTransform, time_d

# Load all necessary data
base_path = '/Users/ashwingirish/Documents/Project_AreaLaw/inspiral_analysis/data_files/'

with open(base_path + 'events_and_detectors.pkl', 'rb') as f:
    event_details = pickle.load(f)

with open(base_path + 'fixed_parameters.pkl', 'rb') as f:
    fixed_parametrs = pickle.load(f)

with open(base_path + 'peak_time.pkl', 'rb') as f:
    peak_time = pickle.load(f)

with open(base_path + 'priors.pkl', 'rb') as f:
    priors = pickle.load(f)

with open(base_path + 'data_dict.pkl', 'rb') as f:
    gw_data = pickle.load(f)

with open(base_path + 'final_mass_and_snr.pkl', 'rb') as f:
    final_mass_snr = pickle.load(f)

# Set up event info
event = "GW150914"
detectors = event_details[event]
det_class = gw_data[event]

N = 2 * 4096
cov_inverse = {
    d: np.linalg.inv(toeplitz(det_class[d]['acf'][0:N]))
    for d in detectors
}

ref_det = detectors[0]
tgps = peak_time[event]
time_delay = time_d(detectors, ref_det, fixed_parametrs[event]['ra'],
                    fixed_parametrs[event]['dec'], tgps)

# Construct likelihood and prior transform
loglikelihood = LogLikelihood(event=event,
                              det_class=det_class,
                              time_delay=time_delay,
                              cov_inverse=cov_inverse,
                              fixed_parametrs=fixed_parametrs,
                              priors=priors,
                              tgps=tgps,
                              N=N,
                              ref_det=ref_det)

prior_transform = PriorTransform(priors, event)

# === Checkpoint handling ===
checkpoint_path = "area_law.save" #This needs to be modified accordingly, make sure you check this
sampler = None

if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
    try:
        with open(checkpoint_path, "rb") as f:
            sampler = dynesty.NestedSampler.restore(checkpoint_path)
        print("Resumed from checkpoint.")
    except Exception as e:
        print(f"Failed to load checkpoint, starting fresh: {e}")
        sampler = None

if sampler is None:
    sampler = dynesty.NestedSampler(
        loglikelihood,
        prior_transform,
        ndim=8,
        sample='unif',
        nlive=100
    )
    print("🚀 Starting fresh Dynesty run.")

# === Run or continue sampling ===
sampler.run_nested(
    checkpoint_file=checkpoint_path,
    checkpoint_every=10,
    resume=True,
    maxiter = 100
)


# Save results
output_file = f'/Users/ashwingirish/Documents/Project_AreaLaw/inspiral_analysis/{event}_results1.pkl'
with open(output_file, 'wb') as f:
    dill.dump(sampler.results, f)

print("Sampling complete.")
