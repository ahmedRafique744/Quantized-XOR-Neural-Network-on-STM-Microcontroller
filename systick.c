#include "systick.h"

static void(*systick_cbfn)(void) = 0;

// Implement a 2 Hz SysTick timer event
void systick_init(void(*cbfn)(void)) {
    // register the callback function
    systick_cbfn = cbfn;

    // Configure a 2 Hz SysTick event by dividing HCLK/8 = 6 MHz by 
    // 3 million.  Note we have to subtract 1 from the desired divisor
    // per the STM32F042K6 programming manual section 4.4.2 
    // register description
    // CHECKPOINT 2 UPDATE: Exception modified to be requested after every 100 ms = 10 Hz (since 6e6/6e5 = 10 Hz)
    // This was computed as follows: 1/t = f --> 1/(100e-3) = 10 Hz. Thus for the divisor, x, 6e6/x = 10 --> x = 6e6/10 -->600000 or 6e5
    SYSTICK->RVR = 600000-1;
    // Switch to the "external clock source" (HCLK/8 = 6 MHz), 
    // enable the counter,
    // and enable an exception request when the counter reaches 0
    SYSTICK->CSR = (SYSTICK_CSR_ENABLE | SYSTICK_CSR_TICKINT);
}

void SysTick_Handler(void) {
    if( systick_cbfn) 
        systick_cbfn();
}