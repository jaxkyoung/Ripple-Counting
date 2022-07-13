/*
 Name:		Ripple_Arduino.ino
 Created:	6/29/2022 2:48:55 PM
 Author:	JYOUNG16
*/

// the setup function runs once when you press reset or power the board


#include <movingAvg.h>
#include <elapsedMillis.h>
#include <Arduino_FreeRTOS.h>

const int LCF_CURR_BUFF_1_SIZE_Z_Pt = 5;

// define tasks
void TaskBlink();
void TaskPeriodicReadCurrent();
void TaskPeriodicFilterCurrent();
void TaskPeriodicPinchThdCalc();
void TaskPeriodicPinchDetection();
void TaskEventProcessSwitchPress();
void TaskPeriodicMotorMainLogic();
void TaskEventActuateMotor();
void TaskPeriodicCalcPwm();

// globals
float rawCurrent_I;
bool postnCnt_B;
float avgCurrent_I;
float currBuff1Avg_I_Calc;
float currBuff2Avg_I_Calc;
float currPinchThd_I;
bool pinchDetct_B;
bool upSwitch_B;
bool downSwitch_B;
bool upReq_B;
bool downReq_B;

int buff1Cnt_Z = 0;
float buff1_ary[5] = { 0 };

elapsedMillis timeStep;

// the setup function runs once when you press reset or power the board
void setup() {
    // initialize serial communication at 9600 bits per second:
    Serial.begin(9600);
    while (!Serial) {
        ; // wait for serial port to connect. Needed for native USB, on LEONARDO, MICRO, YUN, and other 32u4 based boards.
    }
    // Now set up two tasks to run independently.
    // Now the task scheduler, which takes over control of scheduling individual tasks, is automatically started.
    
}

void loop() {
    if (timeStep % 1 == 0) {
        TaskPeriodicReadCurrent();
    }
    if (timeStep >= 10) {

        Buff1MovingAvgFunc();
        Serial.println(currBuff1Avg_I_Calc);
        timeStep = 0;
    }
}

/*--------------------------------------------------*/
/*---------------------- Tasks ---------------------*/
/*--------------------------------------------------*/


void TaskPeriodicReadCurrent() {  // This is a task.
    // read the input on analog pin 0: 
    rawCurrent_I = analogRead(A0);
    // print out the value you read:
    //Serial.println(rawCurrent_I);

}



void Buff1MovingAvgFunc() {
    buff1_ary[buff1Cnt_Z] = rawCurrent_I;
    float buff1Sum_I_Calc = 0;

    for (int i = 0; i < LCF_CURR_BUFF_1_SIZE_Z_Pt; i++) {
        buff1Sum_I_Calc = buff1Sum_I_Calc + buff1_ary[i];
    }
    currBuff1Avg_I_Calc = buff1Sum_I_Calc / LCF_CURR_BUFF_1_SIZE_Z_Pt;
    if (buff1Cnt_Z == LCF_CURR_BUFF_1_SIZE_Z_Pt) { buff1Cnt_Z = 0; }
    else { buff1Cnt_Z++; }
}