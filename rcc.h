#pragma once

#include <stdint.h>

/* Register map definitions from the Reference Manual section 6.4.15 */
typedef volatile struct RCC_CFG_s {
    uint32_t SW       : 2;
    uint32_t SWS      : 2;
    uint32_t HPRE     : 4;
    uint32_t PPRE     : 3;
    uint32_t RES      : 3;
    uint32_t ADCPRE   : 1;
    uint32_t PLLSRC   : 2;
    uint32_t PLLXTPRE : 1;
    uint32_t PLLMUL   : 4;
    uint32_t RES2     : 2;
    uint32_t MCO      : 4;
    uint32_t MCOPRE   : 3;
    uint32_t PLLNODIV : 1;
} rcc_cfg_t;

/* Register map definitions from the Reference Manual section 6.4.15 */
typedef volatile struct RCC_s {
    uint32_t CR;        // 0x0000 Clock Control 
    rcc_cfg_t CFG;      // 0x0004 Clock Config 
    uint32_t CIR;       // 0x0008 Clock Interrupt 
    uint32_t APB2RSTR;  // 0x000C APB2 Reset 
    uint32_t APB1RSTR;  // 0x0010 APB1 Reset
    uint32_t AHBENR;    // 0x0014 AHB Clock Enable 
    uint32_t APB2ENR;   // 0x0018 APB2 Clock Enable
    uint32_t APB1ENR;   // 0x001C APB1 Clock Enable
    uint32_t BDCR;      // 0x0020 RTC Domain Control
    uint32_t CSR;       // 0x0024 Control/Status
    uint32_t AHBRSTR;   // 0x0028 AHB Reset
    uint32_t CFGR2;     // 0x002C Clock Config 2
    uint32_t CFGR3;     // 0x0030 Clock Config 3
    uint32_t CR2;       // 0x0034 Control Register 2
} RCC_t;

// RCC_CFG Config Register Definitions
// Reference Manual Section 6.4.2
#define RCC_CFG_SW_HSI       (0)
#define RCC_CFG_SW_HSE       (1)
#define RCC_CFG_SW_PLL       (2)
#define RCC_CFG_SW_HSI48     (3)

// RCC_APB2RSTR Config Register Definitions
#define RCC_APB2RSTR_ADC_RST  (0x00000200)

// RCC_APB1RSTR Config Register Definitions
// Reference Manual Section 6.4.5
#define RCC_APB1RSTR_TIM2_RST (0x00000001)
#define RCC_APB1RSTR_TIM3_RST (0x00000002)
#define RCC_APB1RSTR_USART2_RST (0x00020000)

// RCC_AHBENR Enable Bit Definitions
// Reference Manual Section 6.4.6
#define RCC_AHBENR_SRAM_EN    (0x00000004)  // SRAM memory clock enable
#define RCC_AHBENR_FLITF_EN   (0x00000010)  // Flash memory clock enable

// Reference Manual Section 6.4.6
#define RCC_AHBENR_IOPA_EN    (0x00020000)
#define RCC_AHBENR_IOPB_EN    (0x00040000)

// RCC_APB2ENR Enable Bit Definitions 
// Reference Manual Section 6.4.7
#define RCC_APB2ENR_ADC_EN    (0x00000200)  // Analog to digital converter clock enable
#define RCC_APB2ENR_USART6_EN (0x00000020)  // USART 6 clock enable
#define RCC_APB2ENR_USART7_EN (0x00000040)  // USART 7 clock enable
#define RCC_APB2ENR_USART8_EN (0x00000080)  // USART 8 clock enable
#define RCC_APB2ENR_USART1_EN (0x00004000)  // USART 1 clock enable

// RCC_APB1ENR Enable Bit Definitions
// Reference Manual Section 6.4.8
#define RCC_APB1ENR_TIM2      (0x00000001)  // Timer 2 clock enable
#define RCC_APB1ENR_TIM3      (0x00000002)  // Timer 3 clock enable
#define RCC_APB1ENR_USART2_EN (0x00020000)  // USART 2 clock enable
#define RCC_APB1ENR_USART3_EN (0x00040000)  // USART 3 clock enable
#define RCC_APB1ENR_USART4_EN (0x00080000)  // USART 4 clock enable
#define RCC_APB1ENR_USART5_EN (0x00100000)  // USART 5 clock enable

// RCC_CR2 Enable Bit Definitions
// Reference Manual Section 6.4.14
// ADC (HSI14) clock enable in CR2
#define RCC_CR2_HSI14_ENABLE  (0x00000001)   // Enable the internal 14 MHz ADC clock
#define RCC_CR2_HSI14_RDY     (0x00000002)   // Ready bit - 14 MHz ADC clock ready
// 48 MHz RC oscillator in CR2
#define RCC_CR2_HSI48ON       (0x00010000)    // Enable the internal 48 MHz system clock
#define RCC_CR2_HSI48RDY      (0x00020000)    // Ready bit - 48 MHz clock ready