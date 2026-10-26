**9/18 - Machine Shop Conversation**

Met with Gregg at the machine shop to get advice on the physical housing and PCB layout:

Board Layout: Keep PCB footprint around 4" x 4" with standard 1/8" mounting holes. Put all I/O connectors on the same edge to keep wiring tidy.
Ports & Controls: Use remote/panel-mount buttons and a remote USB plug so we don't have to align board components directly against the enclosure walls.
Enclosure: Since the shop doesn't do 3D printing and machining an animal shape is impractical, use a stuffed animal plushie to hold the core electronics box.
Feedback & Lighting: Add a simple buzzer and vibration motor for quick haptic/audio alerts (like an alarm clock). Place LEDs along the bottom for an underglow look rather than bare front-facing LEDs.
Next Steps:

Update KiCad outline to 4" x 4" with 1/8" corner holes and single-edge I/O grouping.
Source panel-mount buttons, a remote USB extension, and a suitable plushie.


**10/16 - Design Review Feedback and Improvements**

During our design review, we received feedback that our project needed more technical complexity. Since our hardware design is relatively straightforward and we are all Computer Engineering students, we decided to focus on improving the software side of our project.

After discussing possible improvements, we proposed three changes to Professor Gruev:

Supervised Machine Learning: Replace our hardcoded weighted focus score with a trained Random Forest model. We plan to collect our own sensor data and self-reported focus labels to train and evaluate the model.

Concurrent C++ Architecture: Separate computer vision, OS telemetry, and USB communication into different threads to reduce delays and improve system responsiveness.

GPU-Accelerated Computer Vision: Use GPU processing for more advanced attention tracking, including head-pose estimation, eye tracking, and blink detection (PERCLOS), while maintaining CPU usage below 10%.

Professor's Feedback

Professor Gruev recommended prioritizing supervised ML and GPU-accelerated computer vision, since these would add the most meaningful technical complexity. He explained that simply using existing libraries would not be enough. We should demonstrate our own implementation decisions, optimization, and quantitative testing.

For ML, he suggested comparing our trained model against the original weighted scoring approach using proper training and validation methods. For GPU vision, we should measure accuracy, processing latency, and CPU utilization. He also noted that multithreading would be useful but is more of an architectural improvement unless we can demonstrate measurable performance gains.

He also reminded us that our custom PCB should serve an important function in the project and be properly integrated with our software.

Breadboard Demo Troubleshooting

During our breadboard demonstration, our circuit initially failed to operate as expected. We spent time troubleshooting the ESP32, servo motor, and control signals to identify the problem. Although we could not demonstrate the full functionality during the demo, we were able to diagnose the issue afterward.

Professor Gruev confirmed that we would receive 7.5 points of partial credit for our troubleshooting efforts.

Next Steps

Based on the feedback, we plan to focus on developing an ML-based focus scoring system and improving our computer vision pipeline. We will collect training data, evaluate our model against the original scoring equation, and benchmark CPU usage and processing latency.

Takeaway: adding features alone does not necessarily increase engineering complexity, we need to quantify improvements and ensure our hardware and software work together reliably as a baseline

**10/18 - Fixing the software package**

Set up Noah’s software on my Mac using a Python virtual environment and installed the missing dependencies. Fixed inconsistent file paths and a missing reset method that would crash live inference. Updated the setup instructions and tested the camera, eye tracking, active-app detection, keyboard tracking, and focus predictions over two cycles and recorded those as part of training data. Windows compatibility fixes were included but still need testing with noah. Pushed the changes to michael-dev.

**10/22 - Machine Shop Conversation**
Date: October 20, 2026

1. Enclosure and PCB

we discussed the enclosure we plan to use, which measures approximately 7.2 × 5 × 2.2 inches.

The machine shop said the enclosure should work as long as the PCB and other components fit inside.

They reminded me that these are external dimensions, so I need to account for approximately 1–2 mm of wall thickness.

showed them the PCB layout, and they said the mounting holes looked fine from a mechanical standpoint. However, they couldn't verify the electrical design.

We sent the enclosure product link to the machine shop to order

2. Screen Placement

We discussed whether the screen should be mounted on the removable lid or the main enclosure body.

recommended prioritizing accessibility, since we'll likely need to troubleshoot the PCB more often than the screen.

We discussed keeping the screen fixed so we can remove the lid and easily access the PCB without disturbing the display.

The screen can potentially be glued into place after the necessary opening is cut into the enclosure.

3. Ultrasonic Sensor and Servo

I explained that the ultrasonic sensor would be placed on top of the stuffed animal to detect when someone moves their hand over it, simulating petting.

The sensor will connect to the PCB through wires rather than being mounted directly onto the board.

I also explained that the servo will control a tail-wagging mechanism.

Since our previous servo stopped working, I need to order a replacement and test the mechanism.

4. Screen Shipping and Project Timeline

I mentioned that our screen was being shipped from overseas, with an estimated delivery time of around 10 days.

The technician warned that overseas delivery estimates can be unreliable, especially without confirmed tracking information.

I found a similar screen on Amazon that costs more but could arrive sooner. I plan to look into ordering it as a backup.

The machine shop also has a large backlog of research and senior design projects, so they may not be able to start immediately after our parts arrive.

The technician emphasized that if our screen arrives too late, there may not be enough time to finish fabrication before the demo.

5. Next Steps

Send the enclosure link to the machine shop for ordering.

Confirm the PCB fits inside the enclosure.

Order a replacement servo.

Look into purchasing the screen from Amazon.

Finalize the screen mounting position and tail-wagging mechanism.

Bring all physical components to the machine shop so we can begin fitting and fabricating the enclosure.


