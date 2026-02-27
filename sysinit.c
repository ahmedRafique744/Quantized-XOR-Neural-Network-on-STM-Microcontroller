#include "gpio.h"
#include "rcc.h"

#include "adc.h"
#include "gpio.h"
#include "rcc.h"

#include <stdio.h>

#include "stm32f042k6.h"

// Switch to the 48MHz internal oscillator
static void clock_init(void) {
    // Enable the oscillator (reference manual section 6.4.14)
    RCC->CR2 |= RCC_CR2_HSI48ON;
    // Wait for it to stabilize (reference manual section 6.4.14)
    while( !(RCC->CR2 & RCC_CR2_HSI48RDY) );
    // Switch to using the HSI48 clock source for SYSCLK/HCLK 
    // (reference manual section 6.4.2)
    RCC->CFG.SW = RCC_CFG_SW_HSI48;
}

// Configure General Purpose IO (e.g. the LED on Port B pin 3)
// NOTE that only GPIO clocks for general purpose IO are enabled
// here.  "GPIO peripheral" clocks and port configurations for
// other periperals (e.g. USART, Timer, ADC) are managed in 
// their respective *_init() functions
static void rcc_gpio_init(void) {
    // TODO: comment
    RCC->AHBENR |= RCC_AHBENR_IOPA_EN;
    RCC->AHBENR |= RCC_AHBENR_IOPB_EN;

    // TODO: comment 
    GPIOA->MODER.moder3 = gpio_mode_output;
    GPIOA->MODER.moder4 = gpio_mode_output;

    // TODO: comment
    GPIOB->MODER.moder3 = gpio_mode_output;
}

// Enable USART2 and configure to use Port A pins 2 (TX) and 15 (RX)
static void rcc_usart2_init(void) {
    // Enable the USART2 Communications Interface Peripheral's Clock 
    RCC->APB1ENR |= RCC_APB1ENR_USART2_EN;
    // Reset the USART2 peripheral to out-of-reset defaults
    RCC->APB1RSTR |= RCC_APB1RSTR_USART2_RST;
    RCC->APB1RSTR &= ~RCC_APB1RSTR_USART2_RST;

    // TODO: comment the following
    RCC->AHBENR |= RCC_AHBENR_IOPA_EN;
    GPIOA->MODER.moder2 = gpio_mode_alternate_function;  // USART TX
    GPIOA->AFRL.afsel2 = 1; // USART2 alternate function TX
    GPIOA->MODER.moder15 = gpio_mode_alternate_function; // USART RX
    GPIOA->AFRH.afsel15 = 1; // USART2 alternate function RX
}

// Enable the ADC and configure Port A pin 1 (ADC_IN1) as analog input
static void rcc_adc_init(void) {
    // Enable and reset the ADC peripheral
    RCC->APB2ENR |= RCC_APB2ENR_ADC_EN;
    RCC->APB2RSTR |= RCC_APB2RSTR_ADC_RST;
    RCC->APB2RSTR &= ~RCC_APB2RSTR_ADC_RST;

    // Enable the HSI14 ADC dedicated sampling clock and wait for it to be ready
    // ADC default is to use the HSI14 (ADC) sampling clock
    RCC->CR2 |= RCC_CR2_HSI14_ENABLE;
    while( !(RCC->CR2 & RCC_CR2_HSI14_RDY) );

    // TODO: comment the following
    RCC->AHBENR |= RCC_AHBENR_IOPA_EN;
    GPIOA->MODER.moder0 = gpio_mode_analog;
    GPIOA->MODER.moder1 = gpio_mode_analog;
}

// Initialize system out of reset
void sys_init(void) {
    clock_init();   
    rcc_gpio_init();
    rcc_usart2_init();
    rcc_adc_init();
}