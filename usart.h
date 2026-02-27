#pragma once

#include <stdint.h>

#include "retval.h"

typedef volatile struct USART_s {
    uint32_t CR1;         // 0x00 Config 1
    uint32_t CR2;         // 0x04 Config 2
    uint32_t CR3;         // 0x08 Config 3
    uint32_t BRR;         // 0x0C Baud Rate
    uint32_t GPTR;        // 0x10
    uint32_t RTOR;        // 0x14
    uint32_t RQR;         // 0x18
    uint32_t ISR;         // 0x1C
    uint32_t ICR;         // 0x20
    uint32_t RDR;         // 0x24 Received Data
    uint32_t TDR;         // 0x28 Transmit Data
} USART_t;

#define USART_CR1_UE        (0x00000001) // USART enable
#define USART_CR1_RE        (0x00000004) // USART receiver enable
#define USART_CR1_TE        (0x00000008) // USART transmitter enable
#define USART_CR1_RXNEIE    (0x00000020) // Receiver not empty interrupt enable
#define USART_CR1_TCIE      (0x00000040) // Transmission Complete interrupt enable

#define USART_CR3_DMAT      (0x00000080) // Use DMA for transmission

#define USART_ICR_TCCF      (0x00000040) // Clear USART TC flag in ISR

#define USART_ISR_TXE       (0x00000080) // USART transmit data register empty
#define USART_ISR_TC        (0x00000040) // USART transmission complete
#define USART_ISR_RXNE      (0x00000020) // USART receive register not empty
#define USART_ISR_ORE       (0x00000008) // USART receive overrun

#define USART_ICR_ORECF     (0x00000008) // USART receive overrun clear flag


/// configure usart2 for 115200 baud, 8 data bits, no parity, 1 stop bit.
void usart2_init(void(*usart_rx_cbfn)(uint8_t rx_data));

// Transmit a byte of data
void usart2_tx( uint8_t tx_data );