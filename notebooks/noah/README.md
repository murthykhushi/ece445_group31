# Noah's Lab Notebook

## 11/9

### Objectives:
- Update firmware to display facial expressions instead of plain text
- Restore focus states from proposal/design documents in code (including hysterisis)
- Update data payload to match proposal/design documents
- Added increased petting (ultrasonic sensor) functionality

### What was done:
I focused on integrating the machine-learning pipeline with the microcontroller firmware. I made key changes to the software, creating a new file called `live_inference_test.py`. In it, I focused on reusing the core proposal architecture, which includes the aggregate scoring and smoothing. I also updated the host script to transmit framed ASCII packets over USB serial every 2000 ms matching the proposal format: `$GF,<state_id>,<score>*<XOR_checksum>\n`

I also worked on the firmware by updating a PlatformIO project's `main.cpp` to update the display without full screen redraws. To satisfy this, I implemented off-screen RAM buffers (`TFT_eSprite`). Delta updates push only modified bounding boxes (eyes, mouth/cheeks, HUD text) in ~18 ms.

The petting system also got implemented using the previously integrated HC-SR04 ultrasonic sensor. I added a bi-directional petting protocol where when petted, the device locks the happy `^ ^` eyes open (suspending blinking), flashes the LED status indicator on GPIO 25, and streams an upstream telemetry event (`EVENT:PET_BOOST:<pts>`) over USB serial to `live_inference_test.py`. The host script then reads the petting boost non-blockingly, increases smoothed_score accordingly, and transmits an updated state packet back down to the board.

To test the servo motor functionality, I utilized the small LED setup on GPIO 25. My configuration consisted of a Windows device connected over USB-C COM3 port at 115200 baud.

UPDATE

Before leaving the lab session, Michael was able to bring over the servo we were planning to use. After identifying that we needed separate 5V and 3.3V rails, I identified the LM3940 1-A Low-Dropout Regulator (available in self-service) for our purposes, and added wiring and capacitors to implement the physical servo.

## 10/20

### Objectives:
- Meet as a team to discuss work distribution and POA for the next week
- Complete any work as discussed during the team meeting

### What was done:
We met as a team to talk about what work we needed to do over the next week. Michael would help out with any software as needed like finetuning but focus on making updates to the PCB, Khushi would continue working in the repository she started, and I would add Windows device compatibility and test locally. 

As I began testing, I noticed a few things were off, such as the OS compatibility issue. I solved that by checking the `sys` format and downloading the respective package and using it based on the check. The next issue I found was an `AttributeError: module 'mediapipe' has no attribute 'solutions'`. I discovered that the code was using an outdated package via the Google AI Edge API. By reading through documentation at the [Google API Docs](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python), I was able to update the existing code and successfully ran the program locally.

## 10/12

### Objectives:
- Build a working Servo demo using an ESP32 as a placeholder

### What was done:
By making use of the borrowed ESP32 development board, tactile button, and servo motor, we put together a small working demo using a Python Script and Arduino project. The biggest issue we ran into was connecting the personal device (laptop) to the ESP32. After a short while, we found out that we needed to use a Mac device as opposed to Windows due to a small port error. By making the switch, running a local Python script, and loading the code to the ESP32, we were able to get a small demo of facial detection stopping the servo, a lack of presence to start it, and the button to snooze the servo motor.

## 10/11

### Objectives:
- Create plan of action for upcoming breadboard demo
- Pick up what we can from our order

### What was done:
On Thursday, Michael finalized the PCB parts and order with our TA Tim. Our order was then confirmed and the E-Shop parts were delivered to the lab.
In the lab:
- We organized our parts by what we can use on breadboard and what we couldn't (eg. sorting the capactiors, resistors, LED, and button)
- Found equivalent parts from our old ECE kits
- Built a button for demonstration. Only subsystem we can build with our current parts.

## 10/6

### Objectives:
- Determine which parts need to be ordered
- Delegate work to complete upcoming work for breadboard demo
- Delegate section parts for design review presentation

### What was done:
Michael shared an excel spreadsheet of the parts that can be ordered from the E-Shop. What still needs to be specially ordered is the display, servo, and sensors. He would also finish the remaining work on the PCB before running it through the PCBWay audit. Sections were selected for the presentation.

## 9/29

### Objectives:
- Complete work on PCB KiCad schematics and board design
- Configure STM32 microcontroller based on schematics
- Work on Design Document

### What was done:
Michael worked on designing the PCB schematic and board design. After we reviewed and were satisfied with the schematics that he provided, I began to work on completing the STM32 microcontroller project setup. This included setting up pins, configuring clocks and GPIOs, and ensuring that the generated code was accurate to our design.

The SYSCLK is derived from the 8MHz crystal oscillator. To achieve the 48MHz frequency required for USB operation, the 8MHz clock is routed through the PLL and multiplied by 6.

The servo requires a 50MHz (20ms period) PWM signal, which is derived from TIM2. TIM2 is derived from the 48MHz clock, so the prescalar, auto-reload register (ARR), and pulse needed to be configured. To achieve the 1 microsecond timer tick, the 48MHz clock is divided by 48 and since the prescalar is zero-indexed, the value is set to 47. Now with a 1 microsecond tick, generating a 20ms period requires 20,000 ticks, so ARR is set to 19,999 (again, zero-indexed). To center a servo at 90 degrees, a 1.5ms pulse is required, so pulse value is set at 1500.
