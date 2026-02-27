#pragma once

#include "gpio.h"
#include "rcc.h"
#include "usart.h"

// Memory Map
// Create an instance of the interface on ADC
#define ADC    ((ADC_t *)0x40012400) 
#define RCC    ((RCC_t *) 0x40021000)
#define GPIOA  ((GPIO_t *)0x48000000)
#define GPIOB  ((GPIO_t *)0x48000400)
#define TIM2   ((TIM23_t *)0x40000000)
#define TIM3   ((TIM23_t *)0x40000400)
#define USART1 ((USART_t *)0x40013800)
#define USART2 ((USART_t *)0x40004400)
#define USART3 ((USART_t *)0x40004800)
#define USART4 ((USART_t *)0x40004C00)
#define USART5 ((USART_t *)0x40005000)
#define USART6 ((USART_t *)0x40011400)
#define USART7 ((USART_t *)0x40011800)
#define USART8 ((USART_t *)0x40013800)
