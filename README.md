# Admittance Control: learn by building

Goal: understand admittance control by building it on a 1-DOF MuJoCo "leg".

Core idea: the motor is a stiff **position-controlled** plant (it can't be told
"push with X newtons"). Admittance control adds an outer loop that turns
measured **force into motion**: a virtual mass-spring-damper reacts to the
force, and its displacement is subtracted from the position command.

    M_v*xdd + D_v*xd + K_v*x = F_z        (force in, displacement out)
    X_command = X_target - x_v

## Phases (we pause after each one)

| # | Phase | You write | File |
|---|-------|-----------|------|
| 1 | Rigid plant (MuJoCo XML) | model XML | `phase1/leg.xml` |
| 2 | Baseline collision (inner loop) | 500 Hz position loop + logging | `phase2/baseline.py` |
| 3 | Admittance filter (outer loop) | Euler-integrated virtual dynamics | `phase3/` |
| 4 | Close the loop | `X_cmd = X_target - x_v` | `phase4/` |
| 5 | Tune M_v, D_v, K_v | parameter sweep + plots | `phase5/` |

Setup: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
