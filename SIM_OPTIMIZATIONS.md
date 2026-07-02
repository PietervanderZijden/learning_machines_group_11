# Simulation Interface Optimizations

This branch adds efficiency and reliability improvements to the `robobo_interface`
package. These changes reduce simulation overhead, fix a food-collection callback
bug, and make the CoppeliaSim ZMQ connection more robust.

## Changes

### simulation.py

| Area | Before | After |
|---|---|---|
| **Stepping** | `_sim.step()` (fire-and-forget) | `_client.step()` (synchronous ZMQ round-trip) |
| **Timing** | Defaults left to CoppeliaSim | `configure_simulation_timing()` sets 400 ms control step / 5 ms physics substep |
| **IP default** | `0.0.0.0` (bind address, not portable) | `127.0.0.1` (correct loopback) |
| **ZMQ timeouts** | No send/receive timeouts → hangs on disconnect | `RCVTIMEO` / `SNDTIMEO` set to connection timeout |
| **sleep()** | Busy-wait `time.sleep(0.02)` loop | Uses `_client.step()` in stepping mode; graceful return on sim stop |
| **play/pause/stop** | No timeout on state change → infinite hang | 100-iteration poll with explicit error on failure |
| **block()** | `time.sleep(0.02)` poll | `self.sleep(0.002)` — steps physics instead of sleeping |
| **Rendering** | Always enabled | `setBoolParam(display_enabled, False)` for headless speed |
| **Food callback** | Left as CoppeliaSim default → unreliable | `_patch_food_contact_callback()` patches the `sysCall_contact` script |
| **Fast handles** | Motor joints resolved per call | `_initialise_fast_handles()` caches left/right motor joint handles |
| **Stepping mode** | Off by default | Enabled by default (`_stepping_enabled = True`) |

### hardware.py

| Area | Before | After |
|---|---|---|
| **move()** | Single attempt, fails on ROS exception | 3-attempt retry with 0.5 s service re-discovery |
| **set_wheel_speeds()** | N/A | New convenience method for RL fast-path wheel commands |

### \_\_init\_\_.py

`HardwareRobobo` is now lazily imported to avoid pulling in `rospy` and other
hardware-only dependencies when only the simulation interface is needed.

## Usage

No API changes. Existing code continues to work. The main visible differences are:

- The simulation starts in stepping mode by default.
- `configure_simulation_timing()` is called automatically on connect, so the
  scene must be stopped when `SimulationRobobo()` is constructed.
- `sleep()` uses synchronous stepping when stepping mode is active, so
  simulated time advances accurately.
