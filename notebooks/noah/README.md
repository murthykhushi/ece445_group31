# Noah's Lab Notebook

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
