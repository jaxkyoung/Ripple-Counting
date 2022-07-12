/*
 Name:		Ripple_Arduino.ino
 Created:	6/29/2022 2:48:55 PM
 Author:	JYOUNG16
*/

// the setup function runs once when you press reset or power the board

#include <Arduino_FreeRTOS.h>

// define two tasks for Blink & AnalogRead
void TaskBlink(void* pvParameters);
void TaskAnalogRead(void* pvParameters);

// the setup function runs once when you press reset or power the board
void setup() {

    // initialize serial communication at 9600 bits per second:
    Serial.begin(9600);

    while (!Serial) {
        ; // wait for serial port to connect. Needed for native USB, on LEONARDO, MICRO, YUN, and other 32u4 based boards.
    }

    // Now set up two tasks to run independently.
    xTaskCreate(TaskBlink, "Blink", 128, NULL, 2, NULL);
    xTaskCreate(TaskAnalogRead, "AnalogRead", 128, NULL, 1, NULL);

    // Now the task scheduler, which takes over control of scheduling individual tasks, is automatically started.
}

void loop()
{
    // Empty. Things are done in Tasks.
}

/*--------------------------------------------------*/
/*---------------------- Tasks ---------------------*/
/*--------------------------------------------------*/

void TaskBlink(void* pvParameters) {  // This is a task.
    (void)pvParameters;
    // initialize digital LED_BUILTIN on pin 13 as an output.
    pinMode(LED_BUILTIN, OUTPUT);

    while (1) { // A Task shall never return or exit.
        digitalWrite(LED_BUILTIN, HIGH);   // turn the LED on (HIGH is the voltage level)
        Serial.println("On");
        vTaskDelay(2000 / portTICK_PERIOD_MS); // wait for one second
        digitalWrite(LED_BUILTIN, LOW);    // turn the LED off by making the voltage LOW
        Serial.println("Off");
        vTaskDelay(1000 / portTICK_PERIOD_MS); // wait for one second
    }
}

void TaskAnalogRead(void* pvParameters) {  // This is a task.
    
    (void)pvParameters;

    while (1) {
        // read the input on analog pin 0:
        int sensorValue = analogRead(A0);
        // print out the value you read:
        // Serial.println(sensorValue);
        vTaskDelay(1);  // one tick delay (15ms) in between reads for stability
    }
}
