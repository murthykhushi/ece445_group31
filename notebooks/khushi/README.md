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
