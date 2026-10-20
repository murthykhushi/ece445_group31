# Lab Notebook - 10/16/2026

**Topic:** Data Collection, Model Training & Live Iterations

Completed a 2-hour data session capturing three focus states: "locked in," zero focus, and mixed. The logger recorded all 6 features alongside the `focus_score`. The baseline Random Forest model trained on this cleaned data achieved an $R^2$ of 99.9%.

**Live Inference Iterations**
Initial live testing clustered scores too tightly (15-40). I updated `live_inference.py` to use a hybrid scoring system:
1. **ML Score (40%):** Personalized Random Forest prediction; highly responsive to active input.
2. **Logical Base Score (60%):** Heuristic baseline that penalizes high PERCLOS and rewards productive apps (e.g., VS Code) even if keystrokes drop while reading code.
3. **EMA Smoothing:** An Exponential Moving Average filter applied to the blended score to prevent erratic UI and hardware jumps.

**Current Status:** The model now outputs a wider, accurate score distribution. Minor tuning will continue during real-world use.
--------

# Lab Notebook - 10/16/2026

**Topic:** ## Architecture & ML Pipeline Initialization

This portion bridges the core software architecture and machine learning pipeline to transition the desktop companion from a basic heuristic formula to a data-driven Random Forest regression model.

### Core Scripts
* **`data_logger.py`**: This is the file that captures the data for the model. It runs in the background to capture peripheral telemetry and self-reported focus scores via UI pop-ups, saving them to synchronized CSVs.
* **`train_model.py`**: Supervised learning script that ingests the logged CSVs, performs feature encoding, trains the Random Forest Regressor, and saves the serialized model binary.
* **`live_inference.py`**: The live execution script that loads the trained model, smooths the predictions via an EMA filter, and streams the final focus scores over serial to the STM32 microcontroller.

### Sensor Modules
* **`vision_pipeline.py`**: Called by the main scripts to process webcam frames using MediaPipe, calculating Eye Aspect Ratio (EAR) and rolling Percentage of Eye Closure (PERCLOS).
* **`desktop_pipeline.py`**: Called by the main scripts to asynchronously track keystroke volume, mouse clicks, and the active macOS application using background threads.
---

Goals: 
* Following design review feedback, we have decided to come up with additional software complexities for this project so that we have more ownership, originality, and market competetiveness.
* We wanted to come up with 2-3 more ways for us to increase software complexity to our project. 
  
Session Summary: 
* Michael came up with: Supervised Machine Learning (Model Training): Instead of using our current hardcoded weighting equation to calculate the focus score, we plan to train a lightweight supervised machine learning model, such as a Random Forest Regressor. We will collect our own training data by logging our sensor telemetry alongside self-reported ground-truth labels. This adds algorithmic depth to the project while safely keeping the host application under our strict 10% CPU limit.
* Noah came up with: Concurrent C++ Architecture: To improve our software's architectural rigor, we will transition from a single, sequential loop into a concurrent, multi-threaded design. By separating vision ingestion, OS telemetry tracking, and USB packet dispatching into independent asynchronous processes, we can eliminate blocking delays and make the entire system highly reactive without overcomplicating the firmware.
* I came up with: Python GPU-Accelerated Vision Pipeline:To bypass our strict 10% CPU load limit while extracting deeper behavioral metrics, we plan to shift our face and attention tracking from the host CPU to a dedicated GPU acceleration pipeline using CUDA-enabled PyTorch or MediaPipe. Processing video frames directly in GPU memory allows us to extract continuous metrics like 3D head pose and blink dynamics (PERCLOS) at a high framerate without throttling the user's computer. This introduces advanced computer vision and hardware acceleration to our software stack while keeping host CPU utilization near 0%.

  
**Next Steps:**
1. Reach out to Gruev for approval and thoughts on our ideas. 


# Lab Notebook - 10/01/2026

**Topic:** Design Document Visual Aids
**Author:** Khushi Murthy

---

Goals: 
* Create a visual aid graphic for the design doc + future reference
  
Session Summary: 
* Decided where to place the seperate subsystem.
* Button on the top was a big design choice so the user can easily hit the button for snoozing (like a typical alarm clock)
* Graphic:
* <img width="643" height="420" alt="Screenshot 2026-10-06 at 10 57 19 PM" src="https://github.com/user-attachments/assets/af9614c3-9f84-4db1-8f4e-05c3be8244d3" />

  
**Next Steps:**
1. Finish Design Document with group before deadline
