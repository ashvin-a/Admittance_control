"""Sanity checks for phase 1. Reads your XML, never edits it."""
import os
import sys

import mujoco
import numpy as np

path = os.path.join(os.path.dirname(__file__), "leg.xml")
try:
    m = mujoco.MjModel.from_xml_path(path)
except Exception as e:  # noqa: BLE001
    sys.exit(f"FAIL  XML does not load (finish the TODOs?):\n{e}")
d = mujoco.MjData(m)

results = []


def check(name, ok, hint=""):
    results.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"  -> {hint}"))


check("timestep divides 1/500 s",
      abs(round(0.002 / m.opt.timestep) * m.opt.timestep - 0.002) < 1e-9,
      f"timestep={m.opt.timestep}")
check("one joint", m.njnt == 1, f"njnt={m.njnt}")
if m.njnt == 1:
    check("joint is slide", m.jnt_type[0] == mujoco.mjtJoint.mjJNT_SLIDE)
    check("joint axis is vertical", abs(abs(m.jnt_axis[0][2]) - 1) < 1e-9, str(m.jnt_axis[0]))
    check("joint range is limited", bool(m.jnt_limited[0]))
check("one actuator", m.nu == 1, f"nu={m.nu}")
if m.nu == 1:
    kp = m.actuator_gainprm[0][0]
    check("actuator is position-type (bias = -kp*q)",
          m.actuator_biastype[0] == mujoco.mjtBias.mjBIAS_AFFINE
          and abs(m.actuator_biasprm[0][1] + kp) < 1e-9)
    check("kp is high (>= 1000)", kp >= 1000, f"kp={kp}")
    check("ctrl is range-limited", bool(m.actuator_ctrllimited[0]))
check("force sensor named 'fz'",
      mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "fz") >= 0)

if all(results):
    # Free-hang with ctrl = 0 (joint position 0): foot should sit still in the air, Fz ~ 0.
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    fz = d.sensor("fz").data.copy()
    print(f"\nHolding q=0 for 1 s: q={d.qpos[0]:.4f}, fz={fz}")
    check("no contact force while hovering", np.allclose(fz, 0, atol=1e-6),
          "is the foot already touching the floor at q=0?")

    # Command the lowest reachable position and let it settle: it must hit the floor.
    d.ctrl[0] = m.actuator_ctrlrange[0][0]
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    print(f"Commanded ctrl={d.ctrl[0]:.3f}: q={d.qpos[0]:.4f}, fz={d.sensor('fz').data}")
    check("sensor reads force when pressed into floor",
          np.abs(d.sensor("fz").data).max() > 0.1,
          "Fz stayed ~0: is the commanded target actually below the floor? sensor site on the sole?")

print("\nALL PASS" if all(results) else "\nSome checks failed.")
sys.exit(0 if all(results) else 1)
