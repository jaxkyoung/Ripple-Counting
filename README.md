# Design, Development, and Validation of Sensorless DC Motor Control Software in Automotive
## Introduction
Repository storing all final year project information, research, design and source code

## Project Structure

- 0_Ideation
- 1_Research
- 2_Rig
- 3_Software
- 4_Interim_Report
- 5_Final_Report

## Project Description
JLR currently implement seating control software using feedback from hall-effect sensors in the motors to detect positions; per seat, hall-effect sensors add circa £1 of cost. Removing these sensors can save the business a significant amount per unit [CS.5] [CT.9] [CT.10]. I aim to remove the need for hall-effect sensors in the seat motors by using digital signal processing techniques to identify ripples on the trace of current being drawn by a sensorless DC motor. The ripples identified behave in the same manner as a hall-effect sensor so I will be able to count positions in the same way. This removes the need for them and saves the business a cost per seat unit. I will analyse the current implementation to define functional, non-functional and technical requirements to design [CB.7] [SS.2] [ST.3], implement [SS.3] [SS.4] [ST.4] and test [SS.5] [ST.5] a new sensorless DC control solution in the form of C code [CT.3] [SS.1] [SS.5] implemented on a selection of microcontrollers [CS.2] [SS.6]. During the development I will continually perform static and runtime code analysis to ensure my code meets MISRA-C and MAAB guidelines. The project shall be delivered using the agile framework [ST.2], producing small increments over defined sprints to tie in with time constraints of the business [CS.6]. Finally, I will develop a plant model, using Stateflow and Simulink, of the seat motor to deploy onto a hardware-in-the-loop simulator that will allow me to test independently of hardware.

I will require all current documentation for implementation of position counting within the SZM (seat zone module) and requirements for levels of accuracy necessary for accurate position counting and detection. In terms of hardware, I will use my work-provided laptop for all research, design, and development. I will use an Arduino UNO (Or similar, following further research) to deploy the software and test with physical hardware. I will source a seat motor from an existing vehicle or from the test rigs on site. I will ensure I have a motor capable of ripple and hall sensing to benchmark my solution. Once a prototype version of software is complete, I can look to test on an engineering vehicle or labcar (this would need to be booked through the business).

Risks include the following: 

- Due to the high frequency of ripples within the DC current trace, I will need a high specification microcontroller that has an ADC (Analogue-Digital-Convertor) that operates at a high enough frequency to detect these ripples. This risk can be mitigated by identifying a suitable microcontroller prior to deployment. 
- Electrical interference can have a large impact on the quality of the ripple detected on a current trace, this would need to be taken into account when moving to a production vehicle.

Success shall be defined as a verified and validated [CS.5] [CT.5] DC motor control solution that recreates or improves the current solution used at JLR. This could be verified by testing that we can still detect pinch scenarios using ripple counting for position detection. I will compare total ripple counts vs hall counts to see how close the new system can get to our existing solution. The production solution will not be identical to our current solution but should behave in a similar manner.

## Report Generation
### Requirements
```
sudo apt install pandoc

sudo apt install texlive-latex-recommended
```

### Generation
Once installed, the following command can be run to generate a pdf report.

*Ensure you are in the correct directory prior to executing the command*

```
pandoc report.md --bibliography references.bib --csl elsevier-harvard.csl --highlight-style mystyle.theme -o report.pdf
```