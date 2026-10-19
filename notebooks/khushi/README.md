# Lab Notebook - 10/19/2026
**Topic:** Tuning the ML model & Applying it
---

**Summary**: Added more data using the data collection script, which recorded ear,perclos,is_closed,active_app,keystrokes,mouse_clicks,focus_score. When we trained the data, we got goodfeedback and a large distribution of data points, R^2 == 99.9%. However, I had to iterate on the live script 

To solve this, `live_inference.py` uses a **Hybrid Scoring System** that blends machine learning with rule-based heuristics:

1.  **ML Score (40%):** The Random Forest prediction based on the user's personalized training data. Highly responsive to active input.
2.  **Logical Base Score (60%):** A heuristic baseline that prevents observation bias. It applies severe penalties for high PERCLOS/closed eyes and rewards productive active applications (e.g., VS Code) even if keystrokes drop to zero.
3.  **EMA Smoothing:** A 1D Kalman-style filter (Exponential Moving Average) is applied to the final blended score to prevent rapid, erratic jumping in the UI (and eventually, the hardware peripheral).

Before making this change: The model was stuck at scores around 15 - 40. 
After making this change: The model was able to have a wider span of scores, but still needs some tweaking.

# Lab Notebook - 10/17/2026

**Topic:** Focus State Data Collection & Model Training

---

## Objective
To build a dataset capturing various states of cognitive focus and train a machine learning model to classify these states. The goal is to reliably distinguish between states of being completely "locked in" (hyper-focused), taking a break (zero focus), and a mixed/intermediate state.

## Methodology: Data Collection
Completed a deliberate 2-hour data collection session designed to capture a wide variance in focus levels. I intentionally modulated my environment and workflow to cycle through three distinct cognitive states:

*   **Hyper-Focused ("Locked In"):** Intense concentration, deep work, and zero external distractions.
*   **Zero Focus (Breaks):** Intentional disengagement, relaxed state, and unstructured attention.
*   **Mixed Focus:** Casual task execution with standard contextual switching and average attention span. 

## Current Status
*  COMPLETE:  **Data Gathering Complete:** 2 hours of raw telemetry/data successfully recorded and stored.
*  IN PROGRESS: **Phase Transition:** Moving from the data collection phase into the model training phase.

## Next Steps for Model Training
1.  **Data Preprocessing:** Clean the raw data, handle any missing values, and normalize the inputs.
2.  **Labeling & Windowing:** Segment the 2-hour continuous stream into discrete time windows mapped accurately to the specific focus state (Hyper-focused, Zero focus, Mixed).
3.  **Feature Engineering:** Extract relevant statistical or frequency-domain features from the raw data windows to feed into the model.
4.  **Baseline Model Training:** Train an initial classifier (e.g., Random Forest, SVM, or a lightweight Neural Network) to establish baseline accuracy.
5.  **Evaluation & Tuning:** Evaluate the model against a validation split and tune hyperparameters to improve the distinction between the "Mixed" and "Locked In" states.


# Lab Notebook - 10/16/2026

**Topic:** Software & ML Pipeline Initialization
**Author:** Khushi Murthy

**Objectives:** Scaffold the core software architecture and machine learning pipeline components to transition the desktop companion from a heuristic formula to a data-driven Random Forest regression model.

### File Summary
* **`data_logger.py`**: Background script using OS hooks and ESM pop-ups to capture peripheral telemetry and user-labeled focus scores into synchronized CSV datasets.
* **`train_model.py` (Script 1)**: Supervised learning script that aggregates logged CSVs, performs feature encoding, trains a Random Forest Regressor, and serializes the model binary.
* **`train_model.py` (Script 2)**: Supplementary training script variant configured for iterative feature engineering and performance evaluation ($R^2$ and MAE).
* **`live_inference.py`**: Live execution script that loads the trained model, smooths predictions via a 1D Kalman Filter, and streams focus scores over serial to the STM32 microcontroller.

# Lab Notebook - 10/14/2026

**Topic:** Additional Complexity Ideation 
**Author:** Khushi Murthy

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
