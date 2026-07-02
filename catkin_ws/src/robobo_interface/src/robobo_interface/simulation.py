import math
import os
import queue
import signal
import sys
import threading
import time
from typing import Callable, List, NoReturn, Optional, TypeVar

import cv2
import numpy
import zmq
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from numpy.typing import NDArray
from robobo_interface.base import IRobobo
from robobo_interface.datatypes import (
    Acceleration,
    Emotion,
    LedColor,
    LedId,
    Orientation,
    Position,
    SoundEmotion,
    WheelPosition,
)
from robobo_interface.utils import LockedSet

T = TypeVar("T")


class SimulationRobobo(IRobobo):
    'The simulation robot.'

    def __init__(
        self,
        identifier: int = 0,
        api_port: Optional[int] = None,
        ip_adress: Optional[str] = None,
        logger: Callable[[str], None] = print,
        timeout_dur: int = 10,
        rendering_enabled: bool = False,
    ):
        """Connect to a CoppeliaSim Robobo interface."""
        self._id = identifier
        self._logger = logger
        self._timeout_dur = float(timeout_dur)
        self._used_pids: LockedSet[int] = LockedSet()
        self._identifier = f"[{identifier}]"

        if api_port is None:
            api_port = int(os.getenv("COPPELIA_SIM_PORT", "23000"))



        # 0.0.0.0 to connect to the current computer on Linux, with `--net=host`
        # This doesn't work on Windows or MacOS. There, the variable needs to be specified.
        if ip_adress is None:
            ip_adress = os.getenv("COPPELIA_SIM_IP", "127.0.0.1")

        # The RemoteAPIClient waits indefinetly, but I want some way to show an error.
        # It closes the connection when it gets garbage collected, so no need to close
        try:
            self._client = timeout(
                lambda: RemoteAPIClient(host=ip_adress, port=api_port), timeout_dur
            )
        except TimeoutError:
            self._fail_connect(api_port, ip_adress)
        self._client.socket.setsockopt(zmq.RCVTIMEO, int(timeout_dur * 1000))
        self._client.socket.setsockopt(zmq.SNDTIMEO, int(timeout_dur * 1000))

        try:
            self._sim = timeout(lambda: self._client.require("sim"), timeout_dur)
        except TimeoutError:
            self._fail_connect(api_port, ip_adress)



        if self._sim.getSimulationState() != self._sim.simulation_stopped:
            self.stop_simulation()
        self.configure_simulation_timing()

        try:
            self._sim.setBoolParam(
                self._sim.boolparam_display_enabled, rendering_enabled
            )
        except Exception:
            pass

        self._initialise_handles()
        self._patch_food_contact_callback()
        self._initialise_fast_handles()
        self._stepping_enabled = True
        self._logger(f"""Connected to remote CoppeliaSim API server at port {api_port}
            Connected to robot: {self._identifier}""")

    def set_emotion(self, emotion: Emotion) -> None:
        'Show the emotion of the robot on the screen.'
        self._logger(f"The robot shows {emotion.value} on its screen")

    def move(
        self,
        left_speed: int,
        right_speed: int,
        millis: int,
        blockid: Optional[int] = None,
    ) -> int:
        'Move the robot wheels for `millis` time.'
        if not self.is_running():
            raise RuntimeError("Cannot move wheels when simulation is not running")
        if blockid in self._used_pids:
            raise ValueError(f"BlockID {blockid} is already in use: {self._used_pids}")
        blockid = blockid if blockid is not None else self._first_unblocked()
        self._used_pids.add(blockid)

        self._sim.callScriptFunction(
            "moveWheelsByTime",
            self._wheels_script,
            [right_speed, left_speed],
            [millis / 1000.0],
            [self._block_string(blockid)],
            bytearray(),
        )

        return blockid

    def reset_wheels(self) -> None:
        'Allows to reset the wheel encoder positions to 0.'
        if not self.is_running():
            raise RuntimeError("Cannot reset wheels when simulation is not running")
        self._sim.callScriptFunction(
            "resetWheelEncoders",
            self._wheels_script,
            [],
            [],
            [],
            bytearray(),
        )

    def talk(self, message: str) -> None:
        'Let the robot speak.'
        self._logger(f"The robot {self._identifier} says: {message}")

    def play_emotion_sound(self, emotion: SoundEmotion) -> None:
        'Let the robot make an emotion sound.'
        self._logger(f"The robot {self._identifier} makes sound: {emotion.value}")

    def set_led(self, selector: LedId, color: LedColor) -> None:
        'Set the led of the robot.'
        if not self.is_running():
            raise RuntimeError("Cannot set leds when simulation is not running")
        self._sim.callScriptFunction(
            "setLEDColor",
            self._leds_script,
            [],
            [],
            [selector.value, color.value],
            bytearray(),
        )

    def read_irs(self) -> List[Optional[float]]:
        'Returns sensor readings:.'
        ints, _floats, _strings, _buffer = self._sim.callScriptFunction(
            "readAllIRSensor",
            self._ir_script,
            [],
            [],
            [],
            bytearray(),
        )
        return list(ints)

    def read_image_front(self) -> NDArray[numpy.uint8]:
        'Get the image from the front camera as a numpy array in cv2 format.'
        img, [resX, resY] = self._sim.getVisionSensorImg(self._smartphone_camera)
        img = numpy.frombuffer(img, dtype=numpy.uint8).reshape(resY, resX, 3)

        # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
        # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
        # and color format is RGB triplets, whereas OpenCV uses BGR:
        img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
        return img

    def set_phone_pan(
        self, pan_position: int, pan_speed: int, blockid: Optional[int] = None
    ) -> int:
        'Command the robot to move the smartphone holder in the horizontal (pan) axis.'
        if not self.is_running():
            raise RuntimeError("Cannot set phone pan when simulation is not running")
        if blockid in self._used_pids:
            raise ValueError(f"BlockID {blockid} is already in use: {self._used_pids}")
        blockid = blockid if blockid is not None else self._first_unblocked()
        self._used_pids.add(blockid)

        self._sim.callScriptFunction(
            "movePanTo",
            self._pan_motor_script,
            [pan_position, pan_speed],
            [],
            [self._block_string(blockid)],
            bytearray(),
        )

        return blockid

    def read_phone_pan(self) -> int:
        'Get the current pan of the phone.'
        ints, _floats, _strings, _buffer = self._sim.callScriptFunction(
            "readPanPosition",
            self._pan_motor_script,
            [],
            [],
            [],
            bytearray(),
        )
        return int(ints[0])

    def set_phone_tilt(
        self, tilt_position: int, tilt_speed: int, blockid: Optional[int] = None
    ) -> int:
        'Command the robot to move the smartphone holder in the vertical (tilt) axis.'
        if not self.is_running():
            raise RuntimeError("Cannot set phone tilt when simulation is not running")
        if blockid in self._used_pids:
            raise ValueError(f"BlockID {blockid} is already in use: {self._used_pids}")
        blockid = blockid if blockid is not None else self._first_unblocked()
        self._used_pids.add(blockid)

        self._sim.callScriptFunction(
            "moveTiltTo",
            self._tilt_motor_script,
            [tilt_position, tilt_speed],
            [],
            [self._block_string(blockid)],
            bytearray(),
        )

        return blockid

    def read_phone_tilt(self) -> int:
        'Get the current tilt of the phone.'
        ints, _floats, _strings, _buffer = self._sim.callScriptFunction(
            "readTiltPosition",
            self._tilt_motor_script,
            [],
            [],
            [],
            bytearray(),
        )

        return int(ints[0])

    def read_accel(self) -> Acceleration:
        'Get the acceleration of the robot.'
        _ints, floats, _strings, _buffer = self._sim.callScriptFunction(
            "readAccelerationSensor",
            self._smartphone_script,
            [],
            [],
            [],
            bytearray(),
        )
        return Acceleration(*floats)

    def read_orientation(self) -> Orientation:
        'Get the orientation of the robot.'
        _ints, floats, _strings, _buffer = self._sim.callScriptFunction(
            "readOrientationSensor",
            self._smartphone_script,
            [],
            [],
            [],
            bytearray(),
        )
        return Orientation(*floats)

    def read_wheels(self) -> WheelPosition:
        'Get the wheel orientation and speed of the robot.'
        ints, _floats, _strings, _buffer = self._sim.callScriptFunction(
            "readWheels",
            self._wheels_script,
            [],
            [],
            [],
            bytearray(),
        )
        return WheelPosition(*ints)

    def sleep(self, seconds: float) -> None:
        'Block for an amount of time using simulation stepping if enabled,.'
        if not self.is_running():
            raise RuntimeError("Cannot sleep when simulation is not running")
        if self._stepping_enabled:
            dt = self._sim.getSimulationTimeStep()
            ratio = seconds / dt
            nearest = round(ratio)
            steps = max(
                1,
                int(nearest if math.isclose(ratio, nearest, abs_tol=1e-9) else math.ceil(ratio)),
            )
            for _ in range(steps):
                if not self.is_running():
                    return
                try:
                    self._client.step()
                except zmq.ZMQError as exc:
                    raise RuntimeError(
                        "CoppeliaSim stepping timed out or lost its ZMQ connection"
                    ) from exc
                except Exception:
                    if not self.is_running():
                        return
                    raise
        else:
            start_time = self.get_sim_time()
            while self.get_sim_time() - start_time < seconds:
                if not self.is_running():
                    return
                time.sleep(0.002)

    def is_blocked(self, blockid: int) -> bool:
        'See if the robot is currently "blocked", which is to say, performing an action.'
        res = self._sim.getIntProperty(
            self._sim.handle_scene, self._block_string(blockid)
        )
        if res == 0:
            self._used_pids.discard(blockid)
            return False
        else:
            self._used_pids.add(blockid)
            return True

    def block(self):
        'Block untill (only return once) all blocking actions are completed.'
        while any(self.is_blocked(blockid) for blockid in self._used_pids):
            self.sleep(0.002)

    def play_simulation(self):
        'Start the simulation.'
        self._sim.startSimulation()
        self._wait_for_state(self.is_running, "start")

    def pause_simulation(self):
        'Pause the simulation.'
        self._sim.pauseSimulation()
        self._wait_for_state(self.is_paused, "pause")

    def stop_simulation(self):
        'Stop the simulation.'
        self._sim.stopSimulation()
        self._wait_for_state(self.is_stopped, "stop")

    def _wait_for_state(
        self, predicate: Callable[[], bool], transition: str
    ) -> None:
        """Wait for a simulation transition within the configured timeout."""
        deadline = time.monotonic() + self._timeout_dur
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.002)
        raise RuntimeError(
            f"Simulation failed to {transition} within {self._timeout_dur:g} seconds"
        )

    def is_stopped(self) -> bool:
        'Return wether the simulation is stopped.'
        return self._sim.getSimulationState() == self._sim.simulation_stopped

    def is_paused(self) -> bool:
        'Return wether the simulation is stopped.'
        return self._sim.getSimulationState() == self._sim.simulation_paused

    def is_running(self) -> bool:
        'Return wether the simulation is running.'
        # There are 6 different types of running we don't care about
        state = self._sim.getSimulationState()
        return (state != self._sim.simulation_stopped) and (
            state != self._sim.simulation_paused
        )

    def get_sim_time(self) -> float:
        'Get simulation time (in seconds),.'
        return self._sim.getSimulationTime()

    def get_nr_food_collected(self) -> int:
        'Return the amount of food currently collected.'
        ints, _floats, _strings, _buffer = self._sim.callScriptFunction(
            "remote_get_collected_food",
            self._food_script,
            [],
            [],
            [],
            bytearray(),
        )
        return ints[0]

    def get_position(self) -> Position:
        'Get the position of the Robobo (Relative to the world).'
        pos = self._sim.getObjectPosition(self._robobo, self._sim.handle_world)
        return Position(*pos)

    def get_orientation(self) -> Orientation:
        'Get the orientation of the Robobo (Relative to the world).'
        orient = self._sim.getObjectOrientation(self._robobo, self._sim.handle_world)
        return Orientation(*orient)

    def set_position(self, position: Position, orientation: Orientation) -> None:
        'Set the position of the Robobo in the simulation.'
        self._sim.setObjectPosition(self._robobo, [position.x, position.y, position.z])
        self._sim.setObjectOrientation(
            self._robobo, [orientation.yaw, orientation.pitch, orientation.roll]
        )

    def get_base_position(self) -> Position:
        'Get the position of the base to deliver food at.'
        if self._base is None:
            raise AttributeError("Scene does not have a base")

        pos = self._sim.getObjectPosition(self._base, self._sim.handle_world)
        return Position(*pos)

    def base_detects_food(self) -> bool:
        'Get whether the base detects food on top of it.'
        return self._base_food_distance() > 0

    def _base_food_distance(self) -> float:
        'Get the distance between the food and the base,.'
        if self._base is None:
            raise AttributeError("Scene does not have a base")

        if self._food_script is None:
            raise AttributeError("Cannot find any food in the scene")

        _ints, floats, _strings, _buffer = self._sim.callScriptFunction(
            "getFoodDistance",
            self._base_script,
            [],
            [],
            [],
            bytearray(),
        )
        ret = floats[0]
        if ret < 0:
            raise AttributeError("Cannot find any food in the scene")
        return ret

    def _block_string(self, blockid: int) -> str:
        'Return some unique string based on the identifier and the blockid.'
        # This is technically overkill with the new properties instead of the old signals,
        # But re-doing that would require rewriting some stuff that's not really worth it.
        return f"signal.block_{self._id}_{blockid}"

    def _initialise_handles(self) -> None:
        # fmt: off
        self._robobo = self._get_object(f"/Robobo{self._identifier}")
        self._wheels_script = self._get_childscript(f"/Robobo{self._identifier}/Left_Motor")
        self._leds_script = self._get_childscript(f"/Robobo{self._identifier}/Back_L")
        self._ir_script = self._get_childscript(f"/Robobo{self._identifier}/IR_Back_C")
        self._pan_motor_script = self._get_childscript(f"/Robobo{self._identifier}/Pan_Motor")
        self._tilt_motor_script = self._get_childscript(f"/Robobo{self._identifier}/Pan_Motor/Pan_Respondable/Tilt_Motor")
        self._smartphone_script = self._get_childscript(f"/Robobo{self._identifier}/Pan_Motor/Pan_Respondable/Tilt_Motor/Smartphone_Respondable")
        self._smartphone_camera = self._get_object(f"/Robobo{self._identifier}/Pan_Motor/Pan_Respondable/Tilt_Motor/Smartphone_Respondable/Smartphone_camera")
        # fmt: on

        try:
            self._base = self._get_object("/Base")
            self._base_script = self._get_childscript("/Base")
        except AttributeError:
            self._base = None
            self._base_script = None

        try:
            self._food_script = self._get_childscript("/Food")
        except AttributeError:
            self._food_script = None

    def _get_object(self, name: str) -> int:
        # CoppeliaSim is a mess sometimes.
        try:
            ret = self._sim.getObject(name)
        except:
            raise AttributeError(f"Could not find {name} in scene")
        if ret < 0:
            raise AttributeError(f"Could not find {name} in scene")
        return ret

    def _get_childscript(self, name: str) -> int:
        # Call get_object in here, mostly to get better error messages.
        obj_handle = self._get_object(name)
        try:
            ret = self._sim.getScript(self._sim.scripttype_childscript, obj_handle)
        except:
            raise AttributeError(f"Could not find Script of {name} in scene")
        if ret < 0:
            raise AttributeError(f"Could not find Script of {name} in scene")
        return ret

    def _initialise_fast_handles(self) -> None:
        'Cache joint object handles for direct velocity control (RL fast path).'
        self._left_motor_joint = self._get_object(f"/Robobo{self._identifier}/Left_Motor")
        self._right_motor_joint = self._get_object(f"/Robobo{self._identifier}/Right_Motor")

    def configure_simulation_timing(self) -> None:
        'Restore 400 ms control steps with 5 ms internal dynamics.'
        if self._sim.getSimulationState() != self._sim.simulation_stopped:
            raise RuntimeError("simulation timing can only be configured while stopped")
        self._sim.setFloatParam(self._sim.floatparam_simulation_time_step, 0.4)
        self._sim.setFloatParam(self._sim.floatparam_physicstimestep, 0.005)
        self._sim.setBoolParam(self._sim.boolparam_realtime_simulation, False)
        self._sim.setInt32Param(self._sim.intparam_idle_fps, 0)
        simulation_dt = float(self._sim.getSimulationTimeStep())
        dynamics_dt = float(
            self._sim.getFloatParam(self._sim.floatparam_physicstimestep)
        )
        if not math.isclose(simulation_dt, 0.4, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(
                f"CoppeliaSim simulation timestep must be 0.400 s, "
                f"got {simulation_dt:.6f} s"
            )
        if not math.isclose(dynamics_dt, 0.005, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(
                f"CoppeliaSim dynamics timestep must be 0.005 s, "
                f"got {dynamics_dt:.6f} s"
            )
        self._client.setStepping(True)

    def _patch_food_contact_callback(self) -> None:
        'Make food collection independent of contact handle ordering.'
        if self._food_script is None or not self.is_stopped():
            return
        try:
            text = self._sim.getScriptStringParam(
                self._food_script, self._sim.scriptstringparam_text
            )
            robot_handles = set()
            for handle in self._sim.getObjectsInTree(
                self._robobo,
                self._sim.object_shape_type,
                0,
            ):
                try:
                    respondable = self._sim.getObjectInt32Param(
                        handle,
                        self._sim.shapeintparam_respondable,
                    )
                except Exception:
                    respondable = 0
                if respondable:
                    robot_handles.add(handle)
            if not robot_handles:
                raise RuntimeError(
                    "Robobo model has no respondable shapes for food contact"
                )
            handle_entries = ", ".join(
                f"[{handle}] = true" for handle in sorted(robot_handles)
            )
            helper = (
                f"local robobo_contact_handles = {{{handle_entries}}}\n\n"
                "local function belongs_to_robobo(handle)\n"
                "    return robobo_contact_handles[handle] == true\n"
                "end\n\n"
            )
            helper_start = text.find("local function belongs_to_robobo(handle)")
            if helper_start >= 0:
                table_start = text.rfind(
                    "local robobo_contact_handles", 0, helper_start
                )
                helper_end = text.find("\nend\n", helper_start)
                if helper_end >= 0:
                    removal_start = (
                        table_start if table_start >= 0 else helper_start
                    )
                    text = (
                        text[:removal_start]
                        + text[helper_end + len("\nend\n"):]
                    )
            import_line = 'local sim = require("sim")\n'
            if import_line not in text:
                raise RuntimeError("Food script does not import the CoppeliaSim API")
            text = text.replace(import_line, import_line + "\n" + helper, 1)
            old = """    --if h2:startswith("Food") then
    --    print( h1 .. " <- " .. h2)
    --end"""
            text = text.replace(
                "    --h2 = sim.getObjectName(inData.handle2)",
                "    h2 = sim.getObjectName(inData.handle2)",
            )
            if old in text:
                text = text.replace(
                    old,
                    """    if h2:startswith("Food") and belongs_to_robobo(inData.handle1) then
        collect_food(inData.handle2)
    end""",
                )
            text = text.replace(
                """    if h1:startswith("Food") then""",
                """    if h1:startswith("Food") and belongs_to_robobo(inData.handle2) then""",
            ).replace(
                """    if h1:startswith("Food") then
        collect_food(inData.handle1)
    end""",
                """    if h1:startswith("Food") and belongs_to_robobo(inData.handle2) then
        collect_food(inData.handle1)
    end""",
            ).replace(
                """    if h2:startswith("Food") then
        collect_food(inData.handle2)
    end""",
                """    if h2:startswith("Food") and belongs_to_robobo(inData.handle1) then
        collect_food(inData.handle2)
    end""",
            )
            self._sim.setScriptText(self._food_script, text)
            self._logger(
                "Patched Food contact callback to require Robobo-food contact"
            )
        except Exception as exc:
            self._logger(f"Warning: could not patch Food contact callback: {exc}")

    def _robobo_speed_to_rad_s(self, speed: float, duration_s: float) -> float:
        'Convert Robobo -100..100 speed to rad/s for sim.setJointTargetVelocity().'
        if speed == 0.0:
            return 0.0

        sign = 1.0 if speed > 0.0 else -1.0
        v = abs(speed)

        term1 = (
            1.646e-06 * v**3
            + -2.850e-03 * v**2
            + 6.649 * v
            + 5.114e01
        )
        term2 = (
            -2.912e-04 * v**3
            + 4.647e-02 * v**2
            + -1.339 * v
            + -1.225e01
        )

        velocity_deg_s = term1 + term2 / duration_s
        velocity_rad_s = velocity_deg_s * math.pi / 180.0

        return sign * velocity_rad_s

    def set_wheel_speeds(self, left_speed: float, right_speed: float, duration_s: float = 0.4) -> None:
        'Set wheel velocities, converting Robobo -100..100 units to rad/s.'
        left_vel = self._robobo_speed_to_rad_s(left_speed, duration_s)
        right_vel = self._robobo_speed_to_rad_s(right_speed, duration_s)
        self._sim.setJointTargetVelocity(self._left_motor_joint, left_vel)
        self._sim.setJointTargetVelocity(self._right_motor_joint, right_vel)

    def step_simulation(self, steps: int = 1) -> None:
        'Advance simulation by exactly N timesteps.'
        for _ in range(steps):
            self._client.step()

    def _fail_connect(self, api_port: int, ip_adress: str) -> NoReturn:
        self._logger("""CoppeliaSim Api Connection Error
            Failed connecting to remote API server
            Is CoppeliaSim turned on (with the ZMQ remote API available)?

            If not on Linux with --net=host:
            Did you specify the IP adress of your computer in scripts/setup.bash?
            """)
        self._logger(f"Looked for API at port: {api_port} at IP adress: {ip_adress}")
        raise ConnectionError(
            f"Could not connect to CoppeliaSim at {ip_adress}:{api_port}"
        )


# This only works on Unix. Luckily, we are in Docker.
def timeout(func: Callable[[], T], timeout_duration: int = 10) -> T:
    """Run a callable with a bounded wait."""
    result: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        """Execute the callable and publish its result."""
        try:
            result.put((True, func()))
        except BaseException as exc:
            result.put((False, exc))

    threading.Thread(target=invoke, daemon=True).start()
    try:
        succeeded, value = result.get(timeout=timeout_duration)
    except queue.Empty as exc:
        raise TimeoutError(
            f"operation exceeded {timeout_duration:g} seconds"
        ) from exc
    if succeeded:
        return value  # type: ignore[return-value]
    raise value  # type: ignore[misc]


# The API code catches too much, making it hard to quit when failing.
# This is about as agressive as it gets, and should not be used except in containers.
def quit_hard() -> NoReturn:
    os.kill(os.getpid(), signal.SIGKILL)
    # The above is what does it, but the below is to make the type system happy
    sys.exit(1)
