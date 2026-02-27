#include "led.h"

#include "gpio.h"
#include "rcc.h"

#include "stm32f042k6.h"

void led_toggle(led_t led) {
    switch(led) {
        case LED_USER: gpio_pin_toggle(GPIOB, gpio_pin_3); break;
        default: 
    }
}