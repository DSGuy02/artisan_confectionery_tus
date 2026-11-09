"""One-shot Supervisor feeder for the isolated two-cycle Webots test."""

from controller import Supervisor


TIME_STEP = 16
FIRST_PARK_POSITION = [1.25, 0.8, 0.055]
SECOND_INFEED_POSITION = [0.0, 0.4, 0.055]


def move_workpiece(supervisor, def_name, translation):
    node = supervisor.getFromDef(def_name)
    if node is None:
        return False
    node.getField("translation").setSFVec3f(translation)
    node.resetPhysics()
    return True


def main():
    supervisor = Supervisor()
    time_step = int(supervisor.getBasicTimeStep()) or TIME_STEP
    request_receiver = supervisor.getDevice("feeder_request_receiver")
    ready_emitter = supervisor.getDevice("feeder_ready_emitter")

    if request_receiver is None or ready_emitter is None:
        print("[FEEDER ERROR] Request receiver or ready emitter is unavailable.")
        return

    request_receiver.enable(time_step)
    second_item_fed = False

    while supervisor.step(time_step) != -1:
        if request_receiver.getQueueLength() == 0:
            continue

        message = request_receiver.getString()
        request_receiver.nextPacket()

        if message == "FEED_SECOND":
            if second_item_fed:
                print("[FEEDER ERROR] Second chocolate was already supplied.")
                ready_emitter.send(b"FEEDER_ERROR")
                continue

            first_parked = move_workpiece(
                supervisor, "CHOCOLATE_WORKPIECE", FIRST_PARK_POSITION
            )
            second_presented = move_workpiece(
                supervisor, "CHOCOLATE_WORKPIECE_2", SECOND_INFEED_POSITION
            )
            if not first_parked or not second_presented:
                print("[FEEDER ERROR] Could not stage both workpieces.")
                ready_emitter.send(b"FEEDER_ERROR")
                continue

            second_item_fed = True
            ready_emitter.send(b"SECOND_READY")
            print("[FEEDER] Second chocolate ready")
        elif message == "TEST_DONE":
            print("[FEEDER] Two-cycle test complete; stopping feeder controller.")
            return
        else:
            print(f"[FEEDER ERROR] Unexpected request: {message!r}")
            ready_emitter.send(b"FEEDER_ERROR")


if __name__ == "__main__":
    main()