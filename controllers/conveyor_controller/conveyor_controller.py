"""
conveyor_controller.py - Indexed Conveyor Belt & Optical Arrival Gate Controller
Cyberbotics Webots Controller for Infeed Conveyor Track
Technological University of the Shannon (TUS Athlone)
Integrated Robotics 4 - Continuous Assessment 3
"""

from controller import Robot

BELT_SPEED_MPS = 0.12
DETECTION_THRESHOLD_M = 0.11
SETTLE_STEPS_AFTER_STOP = 12
EMPTY_BELT_TIMEOUT_SEC = 4.5


def main():
    robot = Robot()
    time_step = int(robot.getBasicTimeStep()) or 16

    belt_motor = robot.getDevice("belt_motor")
    arrival_sensor = robot.getDevice("arrival_sensor")
    cmd_receiver = robot.getDevice("conveyor_cmd_receiver")
    status_emitter = robot.getDevice("conveyor_status_emitter")

    if not all([belt_motor, arrival_sensor, cmd_receiver, status_emitter]):
        print("[CONVEYOR ERROR] Missing required conveyor device(s).")
        return

    belt_motor.setPosition(float("inf"))
    belt_motor.setVelocity(0.0)
    arrival_sensor.enable(time_step)
    cmd_receiver.enable(time_step)

    state = "STOPPED"
    settle_steps = 0
    advance_start_time = 0.0
    gate_was_clear = False

    print("[CONVEYOR] Indexed conveyor controller initialized (belt locked).")

    while robot.step(time_step) != -1:
        while cmd_receiver.getQueueLength() > 0:
            msg = cmd_receiver.getString()
            cmd_receiver.nextPacket()

            if msg == "ADVANCE_BELT":
                if state == "STOPPED":
                    advance_start_time = robot.getTime()
                    gate_was_clear = arrival_sensor.getValue() >= DETECTION_THRESHOLD_M
                    belt_motor.setVelocity(BELT_SPEED_MPS)
                    state = "ADVANCING"
                    print("[CONVEYOR] Belt advancing at 0.12 m/s; monitoring optical gate...")
            elif msg == "DEMO_DONE":
                belt_motor.setVelocity(0.0)
                print("[CONVEYOR] Full demo complete; stopping conveyor controller.")
                return

        if state == "ADVANCING":
            dist = arrival_sensor.getValue()
            if not gate_was_clear:
                if dist >= DETECTION_THRESHOLD_M:
                    gate_was_clear = True
            elif dist < DETECTION_THRESHOLD_M:
                belt_motor.setVelocity(0.0)
                settle_steps = SETTLE_STEPS_AFTER_STOP
                state = "SETTLING"
                print(f"[CONVEYOR] Optical gate triggered (d={dist:.4f} m). Belt stopped.")
            elif (robot.getTime() - advance_start_time) >= EMPTY_BELT_TIMEOUT_SEC:
                belt_motor.setVelocity(0.0)
                state = "STOPPED"
                status_emitter.send(b"BATCH_COMPLETE")
                print("[CONVEYOR] No further workpieces arrived before timeout; batch complete.")

        elif state == "SETTLING":
            settle_steps -= 1
            if settle_steps <= 0:
                state = "STOPPED"
                status_emitter.send(b"PIECE_READY")
                print("[CONVEYOR] Workpiece settled at inspection zone [0.00, 0.40]; notified supervisor.")


if __name__ == "__main__":
    main()
