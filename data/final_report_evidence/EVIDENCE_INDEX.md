# Partner 2 Final Report Evidence Index

This index records the evidence currently present in the repository for the isolated lost-grip recovery and two-cycle tests. Code documents intended behavior; it is not, by itself, proof that a Webots run completed as intended. Captured multi-cycle and lost-grip Webots Console transcripts are saved at `data/final_report_evidence/multicycle_console.txt` and `data/final_report_evidence/lost_grip_recovery_console.txt`.

## Lost-Grip Recovery

| Existing artifact | What it demonstrates | Limitations |
| --- | --- | --- |
| `controllers/cell_supervisor_grip_test/cell_supervisor_grip_test.py` | Contains a one-shot simulated lost-grip injection at a transfer checkpoint, then routes to `SAFE_RECOVERY`; checks six joint-position sensors for Home. | Source code describes behavior, not a particular executed run. The injection is not a real physical slip. |
| `data/production_metrics.csv` | Contains the matching 9.65-second records `1794247630.18,1,Dark,45.0,1.0,0.0,9.65,Fault_Lost_Grip,ABORT_LOST_GRIP` and `1794263142.95,1,Dark,45.0,1.0,0.0,9.65,Fault_Lost_Grip,ABORT_LOST_GRIP`. | The rows record an abort and cycle duration, but have no Home-verification field or RunID. `GripForceN=0.0` is a logger value, not measured force; the rows alone do not prove `[HOME VERIFIED]`. |
| `controllers/cell_supervisor/cell_supervisor.py` | Shows the production controller's grip checks and original `SAFE_RECOVERY` commands. | The original controller does not verify measured Home joint positions; it uses timed waits and reports completion. It does not contain the isolated injection. |
| `data/final_report_evidence/lost_grip_recovery_console.txt` | Captured Console output shows the one-shot simulated fault injection, transfer abort, recovery commands, `[HOME VERIFIED] Robot reached Home position`, and a 9.65-second cycle. | Demonstrates simulated fault handling and six-joint Home verification in the isolated controller. It does not demonstrate real physical slip detection, calibrated grip force, or measured Cartesian clearance. No RunID links it independently to the CSV row. |

### Evidence Interpretation and Remaining Limits

In the saved lost-grip transcript, touch readings are 0.000 and finger-position readings are reported in radians; neither is a calibrated grip-force measurement. The test keeps the gripper closed and deliberately overrides the grasp result, so it does not demonstrate a real physical slip. The recovery log explicitly says Cartesian clearance is not measured. The transcript's 9.65-second duration matches a CSV row, but without a RunID this is not an independently established row-to-run association.

## Automatic Two-Cycle Test

| Existing artifact | What it demonstrates | Limitations |
| --- | --- | --- |
| `controllers/cell_supervisor_multicycle_test/cell_supervisor_multicycle_test.py` | Implements the isolated `CYCLE_COMPLETE` handshake, waits for `SECOND_READY`, returns to `IDLE`, and stops after Cycle 2. | Code establishes the intended controller sequence, not physical placement. |
| `controllers/cell_supervisor_multicycle_feeder/cell_supervisor_multicycle_feeder.py` | Receives `FEED_SECOND`, parks the first workpiece, moves the distinct second workpiece to the infeed, and sends `SECOND_READY` after the positioning calls succeed. | No saved Console trace confirms those messages occurred in the same run. |
| `worlds/confectionery_cell_multicycle_test.wbt` | Defines separate first and second chocolate nodes, with the second staged outside the camera view, plus the feeder communication devices. | World configuration alone is not runtime evidence. |
| `data/multicycle_test_metrics.csv` | Contains six `CYCLE_SEQUENCE_COMPLETE` records: four CycleID 1 rows and two CycleID 2 rows. The final rows record Dark at +45 degrees and Dark at 0 degrees, respectively. | There is no RunID, so the rows cannot conclusively be paired to a single run. `CYCLE_SEQUENCE_COMPLETE` is a controller status, not independent placement verification. |
| `data/vision_validation/multicycle_test/annotated_camera.png` | Shows one Dark chocolate frame labelled PASS at +45 degrees with solidity 1.000. | It does not show Cycle 2 or the feeder handshake and is not linked to a specific CSV row. |
| `data/vision_validation/multicycle_test/raw_camera.png` | Preserves the raw camera frame associated with the available multi-cycle annotated image. | It documents one frame only; it does not prove two consecutive placements. |
| `data/final_report_evidence/multicycle_console.txt` | Captured Console output shows Cycle 1 at +45 degrees, the feeder request and ready messages, Cycle 2 at 0 degrees, both controller-sequence completion messages, and successful controller exits. | Demonstrates two controller cycles and successful mock-feeder communication, not independently verified physical tray placement. It contains no RunID and is not assigned to particular CSV rows. |

### Evidence Interpretation and Remaining Limits

The saved transcript includes `[FEEDER] Requesting second chocolate.` and `[FEEDER] Second chocolate ready` messages, but does not print the wire payload strings themselves. It states that release was commanded and that placement was not independently sensed. The transcript, CSV and single annotated frame therefore document the controller sequence and communication, not physical placement or a definitive RunID-to-row association.

## Related Visual Evidence

`data/vision_validation/defect_45deg/annotated_camera.png` shows a Dark +45-degree vision result with solidity 0.579 and `REJECT: DEFECT`; it supports the separate defect-inspection example, not lost-grip recovery or multi-cycle execution. The root `data/vision_validation/raw_camera.png` and `annotated_camera.png` currently match the `dark_45deg` (and `multicycle_test`) images and have no run-specific Console context beyond `lost_grip_recovery_console.txt` saving to `data/vision_validation`.