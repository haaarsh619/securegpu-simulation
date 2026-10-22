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
    slo_first.py
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

figures/
    summary_<N>gpus.png              SLO, admission, memory util, compliance (2x2)
    slo_satisfaction_<N>gpus.png     per-metric curves vs. arrival rate,
    admission_rate_<N>gpus.png       mean of seeds with 95% CI band
    memory_utilization_<N>gpus.png
    gpu_busy_time_<N>gpus.png
    queue_delay_<N>gpus.png
    tradeoff_slo_vs_compliance.png   security vs. performance per scheduler
    mode_mix.png                     share of SH / ISO / EX placements
    unsafe_colocation.png            unsafe co-location by GPU count
    gpu_scaling.png                  SLO satisfaction vs. number of GPUs
    slo_by_workload_type.png         inference / fine-tuning / training
    queue_delay_by_workload_type.png
    algorithm1_risk_modes.png        pairwise risk -> required mode
    gantt_timeline.png               placements on each GPU for one run

METRICS
-------
admission_rate = share of jobs placed on a GPU before their deadline.
security_policy_compliance = 1 - unsafe_colocation_rate.

IMPORTANT
---------
Workload service times, compatibility values, and risk thresholds
are controlled simulator inputs unless replaced by actual measured
or trace-derived values.
