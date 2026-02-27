#pragma once

/* An LED type to limit arguments to valid LEDs */
/* LED_USER is the green LED on the development board */
typedef enum {LED_USER} led_t;

/* Toggle an LED */
void led_toggle(led_t led);
