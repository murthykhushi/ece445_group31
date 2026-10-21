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
