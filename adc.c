#include "adc.h"
#include "nvic.h"

#include "stm32f042k6.h"

static ADC_CHANNEL_t active_channel;

// ADC call-back function for interrupt handling.
static void(*adc_cbfn_ptr)(ADC_CHANNEL_t channel, uint16_t data) = 0;

// enable the ADC and any IO pins used by the ADC using the bus clock (8 Mhz)
// configure for 12-bit sampling mode
void adc_init(void(*adc_cbfn)(ADC_CHANNEL_t channel, uint16_t data)) {

    // Register the ADC callback function
    adc_cbfn_ptr = adc_cbfn;

    // Set the correct sampling time for the input signal's impedance
    ADC->SMPR = ADC_SMPR_13_5;

    // Calibrate the ADC
    ADC->CR |= ADC_CR_ADCAL;
    // hardware clears the bit when calibration completes
    // if your program hangs and you hit break & are sitting on this 
    // line something is wrong with your request to calibrate... 
    // check the requirements in the user manual!
    while( ADC->CR & ADC_CR_ADCAL);

    // Enable the ADC
    ADC->CR |= ADC_CR_ADEN;

    // Wait for the ADC to become ready
    while( !(ADC->ISR & ADC_ISR_ADRDY) );

    // CHECKPOINT 3 CODE:
    // Eenabling peripheral interrupts
    ADC->IER = ADC_IER_EOCIE;

    // Enabling ADC Interrupts in the NVIC for propogation to processor
    NVIC_ISER = NVIC_ISER_SETENA_ADC_COMP;
} 

// initiate a conversion from the selected channel
ret_val_t adc_convert(ADC_CHANNEL_t channel) {
    // Make sure the ADC is ready
    if( !(ADC->ISR & ADC_ISR_ADRDY ) )
        return RET_ERROR;

    // Configure the channel selection register 
    ADC->CHSELR = channel;

    // Storing the channel being converted.
    active_channel = channel;

    // Start the conversion
    ADC->CR |= ADC_CR_ADSTART;

    return RET_SUCCESS;
}

// CHECKPOINT 3 - Interrupt Handler Function 
void ADC1_IRQHandler(void){
    if (ADC->ISR & ADC_ISR_EOC) {
        // Reads data register for converted value and clears EOC flag.
        uint16_t data = ADC->DR;

        // Ensures that the ADC call back function is valid and not 0
        if (adc_cbfn_ptr) {
            adc_cbfn_ptr(active_channel, data);
        }
    }
}