#include "usart.h"
#include "stm32f042k6.h"

#include <stdint.h>

#include "nvic.h"

// Registered callback function, if any, for RX'd characters
static void(*usart_rx_cbfn_ptr)(uint8_t data) = 0;

void USART2_IRQHandler(void) {
    // handle receive interrupt 
    if( USART2->ISR & USART_ISR_RXNE ) {
        // check for (and clear) overruns
        if( USART2->ISR & USART_ISR_ORE ) {
            USART2->ICR = USART_ICR_ORECF;
        }
        
        // read received data, clears RXNEIE flag
        uint8_t rx_data = USART2->RDR;
        // if a valid callback function pointer is registered, call it
        if( usart_rx_cbfn_ptr ) {
            usart_rx_cbfn_ptr(rx_data);
        }
    }
}

void usart2_init(void(*usart_rx_cbfn)(uint8_t data)) {

    // Defaults are for 8 data, no parity, 1 stop bit, so just need to set baud rate
    // Using 48 MHz clock for 115200: 
    USART2->BRR = 0x1A1; // from Table 106 Section 27.5.4 of reference manual

    // Register the callback function pointer
    usart_rx_cbfn_ptr = usart_rx_cbfn;

    // Enable USART2 rx interrupts in the USART
    USART2->CR1 |= USART_CR1_RXNEIE;

    // Enable any USART2 interrupts through the NVIC
    NVIC_ISER |= NVIC_ISER_SETENA_USART2;

    // Enable the USART
    USART2->CR1 |= USART_CR1_UE;
    USART2->CR1 |= USART_CR1_RE; // receive enable
    USART2->CR1 |= USART_CR1_TE; // transmit enable
}

// send a byte of data 
void usart2_tx(uint8_t tx_data) {
    // make sure the transmit register is available 
    // (USART is not actively transmitting something else...)
    // if USART2 is busy, wait for it to be available
    while(!(USART2->ISR & USART_ISR_TXE));
    // send the byte
    USART2->TDR = tx_data;
}
