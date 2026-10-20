SecureGPU Simulator
===================

This project implements the simulator algorithm from the
SecureGPU proposal:

1. Security compatibility determines required protection.
2. Resource feasibility filters invalid workload-GPU pairs.
3. SLO slack ranks only the feasible candidates.
4. Lowest pairwise risk is the first tie-breaker.
5. Earliest arrival is the final tie-breaker.

FILES
-----
config.py
workload.py
gpu.py
security.py
generator.py
simulator.py
metrics.py
plot_results.py
main.py
quick_test.py

schedulers/
    base.py
    unrestricted.py
    strict_isolation.py
    security_only.py
    securegpu.py
    __init__.py

INSTALL
-------
pip install -r requirements.txt

QUICK TEST
----------
python quick_test.py

FULL EXPERIMENT
---------------
python main.py

OUTPUTS
-------
securegpu_results.csv
slo_satisfaction.png
unsafe_colocation.png
gpu_utilization.png

IMPORTANT
---------
Workload service times, compatibility values, and risk thresholds
are controlled simulator inputs unless replaced by actual measured
or trace-derived values.
