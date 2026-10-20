# Lab Notebook - 09/18/2026

**Topic:** Machine Shop Notes (Gregg)  
**Author:** Michael Yang
---
Met with Gregg at the machine shop to get advice on the physical housing and PCB layout:

* Board Layout: Keep PCB footprint around 4" x 4" with standard 1/8" mounting holes. Put all I/O connectors on the same edge to keep wiring tidy.
* Ports & Controls: Use remote/panel-mount buttons and a remote USB plug so we don't have to align board components directly against the enclosure walls.
* Enclosure: Since the shop doesn't do 3D printing and machining an animal shape is impractical, use a stuffed animal plushie to hold the core electronics box.
* Feedback & Lighting: Add a simple buzzer and vibration motor for quick haptic/audio alerts (like an alarm clock). Place LEDs along the bottom for an underglow look rather than bare front-facing LEDs.

**Next Steps:**
1. Update KiCad outline to 4" x 4" with 1/8" corner holes and single-edge I/O grouping.
2. Source panel-mount buttons, a remote USB extension, and a suitable plushie.

# Lab Notebook - 09/28/2026

**Topic:** Pre PCB Design Meeting 
**Author:** Khushi Murthy
---
Met with group to discuss PCB layout:

* Designed all essential parts
* <img width="933" height="658" alt="Screenshot 2026-09-29 at 2 33 58 PM" src="https://github.com/user-attachments/assets/cce26a5e-0d58-49a2-9e8a-6ca2e844dc6e" />

**Next Steps:**
1. Go to PCB review Tuesday in ECEB
2. Edit PCB design based on feedback

# Lab Notebook - 09/29/2026

**Topic:** PCB Design Meeting 
**Author:** Khushi Murthy
---
Met with group to discuss PCB layout:
Goals: 
* Review Design
* Prepare for possibly ordering the board

Meeting Summaries
* Added mounting holes 1/8th inch each 
* Consolidated Order Spreadsheet
* Decided a final pcb layout with all parts on one board
* Finished the wiring
 
**Next Steps:**
1. Submit an order 
2. Edit PCB design based on feedback

# Lab Notebook - 10/06/2026

**Topic:** Pre Design Review Meeting 
**Author:** Khushi Murthy
---
Goals: 
* Review design decisions with group
* Ensure we're on pace to submit PCB order by second round of order
* Review the component order through ECE shop


Meeting Summaries & decisions
* Reviewed design decisions like no 3d encasing, two different power railings (3.3V and 5V)
* Debounce circuit for single button clicks to not get mistaken as double click via circuit
* voltage divider from sensing & motors to MCU to go down from 5V to 3.3V
 
**Next Steps:**
1. Design Review Presentation Tomorrow
2. Breadboard work 

# Lab Notebook - 10/07/2026

**Topic:** Post Design Review Meeting 
**Author:** Khushi Murthy
---
Goals: 
* Go over feedback from professor about our design
* Decide on next steps for project and what we want to edit on our project 

Meeting Summaries & decisions
* Professor feedback was that we were lacking complexity in our design --> we decided we wanted to elevate the software complexity as we are all more experienced in that field and have taken less electrical/ hardware classes
* Some ideas we have are:
   * Upgrade the 1.0 Hz OpenCV webcam presence tracking to a custom CUDA and TensorRT GPU vision pipeline that tracks 60 FPS 3D head pose and blink dynamics, ensuring the background host application stays below its 10% CPU limit, which was one of our goals.
   * Transform the local USB serial telemetry, which currently transmits focus scores every 2000 ms, into a distributed system utilizing a custom UDP/QUIC network client and FlatBuffers serialization for low-latency remote synchronization.
   * Transition the STM32 microcontroller from its existing bare-metal four-state behavioral machine to a preemptive FreeRTOS architecture, integrating a custom ARM Thumb-2 assembly decompression routine to render advanced facial animations without exceeding the hardware's 20 KB SRAM limit.  
 
**Next Steps:**
1. Finish the PCB audit
2. Send to Tim for ordering

# Lab Notebook - 10/14/2026

**Topic:** Breadboard Demo Notes
**Author:** Khushi Murthy
---
Demo Summaries: 
* Goal was to have a servo motor running at different speeds based on how focused we were, and an input button that could "snooze" the motor. The snooze would wear off after 10 seconds of continuous facial detection.
* Had a working demo on Monday but during demo itself it was not working.
* Diagnosed that it was because the esp ran on 3.3V while the motor ran on 5V and sometimes could work because threshold was so close.
* We added a SN3904 transistor to step the voltage properly but at this point both motors we had were fried.
* Recieved 7.5/10 points for diagnosing the issue and understanding the circuit 

**Next Steps:**
1. Get a new motor so that we can test software components that we have so far before the PCB board arrives 

# Lab Notebook - 10/15/2026

**Topic:** Design Review Feedback Notes 
**Author:** Khushi Murthy
---

Topic: 
* We got the following feedback from design review and we wanted to address concerns here:

Feedback & Response: 
1) How will we fit the components & PCB board into a pre existing stuffed animal?
   - All electronics and moving parts mount securely to a rigid 3D-printed PETG internal skeleton.
   - Face/Belly: A 3.5-inch TFT display sits flush with the exterior for clear animations.
   - Forehead: Ultrasonic sensors tilt 15° forward through the fabric, sitting next to a tactile push-button.
   - Rear: A servo anchors to the spine to drive the articulated tail.
   - Base: The main PCB, vibration motor, and a weighted stability plate sit at the bottom, featuring a recessed USB-C port.
2) How do we differ from commercial products?
  - Unlike standard on-screen blockers that are easily ignored or muted, our device offers:
  - Physical Presence: A tangible, desk-level companion instead of a dismissible digital pop-up.
  - Gentle Feedback: Real-world cues like tail-wagging and facial expressions rather than stressful alarms or forced screen-locks.
  - Positive Reinforcement: It builds sustainable focus habits without causing "toxic productivity guilt".


# Lab Notebook - 10/16/2026

**Topic:** Updated Weekly Scheduele Post New Additions
**Author:** Khushi Murthy
---

Notebook Entry Description: 
* An Updated weekly scheduele post adding new software complexity to project (ML training and possibly gpu acceleration):

* Week 1: ML Pipeline & Ground Truth Collection**
  * Deploy experience sampling (ESM) script and task-bounded logging to collect 2,500–3,000 feature/label rows.
  * *Deliverable:* Active telemetry dataset logging.

* Week 2: Model Training & Vision Pipeline**
  * Train scikit-learn Random Forest Regressor on PERCLOS, head pose, and typing entropy; evaluate $R^2$ and MAE.
  * Integrate ultrasonic and button inputs on MCU; prototype GPU-accelerated MediaPipe vision as stretch goal.
  * *Deliverable:* Validated focus ML model; sensor communication verified.

* Week 3: Subsystem Integration & Kalman Filter**
  * Assemble PCB 
  * Replace heuristic formula with ML model inference and 1D Kalman state smoothing.
  * Mount PCB, 3.5" TFT display, and HS-311 servo to internal 3D-printed PETG skeletal chassis.
  * *Deliverable:* Live ML focus score streaming over USB under <10% host CPU load.

* Week 4: Full Assembly & Enclosure Fitting**
  * Fit PETG skeleton and electronics inside plush enclosure, routing screen bezel and sensor eyelets.
  * Integrate USB-C host communication with STM32 behavioral state machine.
  * *Deliverable:* Fully assembled physical desktop companion prototype.

* Week 5: End-to-End System Testing & Thermal Validation**
  * Perform R&V table tests: DMA display latency (<100 ms) and servo sweeps (<500 ms).
  * Stress-test AMS1117-3.3 regulator thermal limits inside enclosed plush housing.
  * *Deliverable:* Complete system verification; passing mock demonstration.

* Week 6: Final Presentation & Deliverables**
  * Generate ML validation curves, feature importance charts, and finalize presentation deck.
  * Complete final engineering design report.
  * *Deliverable:* Final Functional Demo.

# Lab Notebook - 10/17/2026

**Topic:** ML Pipeline
**Author:** Khushi Murthy
---

This pipeline uses a machine learning model to predict your real-time cognitive focus on a scale of 0-100. It fuses webcam vision data with OS-level desktop telemetry to drive a physical desk peripheral, replacing the old hardcoded heuristic scoring.

## How it Works
The system aggregates your telemetry into 60-second rolling windows to feed the model:
* **Vision:** Eye Aspect Ratio (EAR), Percentage of Eye Closure (PERCLOS), and a binary check for if your eyes are closed.
* **Desktop:** The currently active application (integer encoded), plus raw keystroke and mouse click counts to differentiate between active work (typing) and passive work (reading/watching).

The pipeline uses a `scikit-learn` Random Forest Regressor. This keeps background CPU usage safely under 10% during live inference and handles the mix of continuous vision data and categorical application data perfectly.

## How to Train the Model
To get the model working, you need to collect ground-truth data so it learns what your specific focus habits look like.

1. **Run the data logger.** The script will run in the background and automatically prompt you every 60 seconds to self-report your current focus score. 
2. **Vary your states.** For a highly accurate model, ensure you log data across different mindsets: high focus (intense coding), medium focus (reading docs), and low focus (casual scrolling, taking a break).
3. **Hit your data targets.** Aim for about 3 hours of logged data (~180 pings) to validate your proof-of-concept. To establish a robust, personalized baseline, you will eventually want to collect 1,000 to 2,000 instances (16-32 hours of background running).

## How to Run Live Inference
Once model is trained and saved, run `live_inference.py`. This script loads your personalized Random Forest model and begins real-time tracking in the background. It predicts your focus every 60 seconds, applies an EMA smoothing filter to prevent erratic jumping, and outputs the final blended score to drive your hardware peripheral.
