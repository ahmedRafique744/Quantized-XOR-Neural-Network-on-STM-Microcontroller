#pragma once

/* NOTE: you should not need to change the contents of this file! */

#include <stdbool.h>
#include <stdint.h>

// Peripheral driver functions always return a ret_val_t of:
//    RET_SUCCESS       	on success
//    RET_UNIMPLEMENTED 	if the function is not implemented
//    RET_ERROR             a generic error occurred
//    ... (< -2)            a function-specific error occurred
//                          (these are defined in the C module implementing
//                          the function, e.g. '#define RET_GPIO_ERR_NULL_PTR')
#include "retval.h"
#define RET_GPIO_ERR_NULL_PTR     (-3)

// The following structure is based on register map definitions from 
// reference manual RM0091 section 8.4.1 and uses C bitfields to enable
// configuration of a single pin's mode
typedef struct gpio_mode_s {
    uint32_t moder0 : 2;
    uint32_t moder1 : 2;
    uint32_t moder2 : 2;
    uint32_t moder3 : 2;
    uint32_t moder4 : 2;
    uint32_t moder5 : 2;
    uint32_t moder6 : 2;
    uint32_t moder7 : 2;
    uint32_t moder8 : 2;
    uint32_t moder9 : 2;
    uint32_t moder10 : 2;
    uint32_t moder11 : 2;
    uint32_t moder12 : 2;
    uint32_t moder13 : 2;
    uint32_t moder14 : 2;
    uint32_t moder15 : 2;
} gpio_mode_t;

// The following structure is based on register map definitions from 
// reference manual RM0091 section 8.4.9 and uses C bitfields to enable
// configuration of a single pin's alternate function 
typedef struct gpio_afrl_s {
    uint32_t afsel0 : 4;
    uint32_t afsel1 : 4;
    uint32_t afsel2 : 4;
    uint32_t afsel3 : 4;
    uint32_t afsel4 : 4;
    uint32_t afsel5 : 4;
    uint32_t afsel6 : 4;
    uint32_t afsel7 : 4;
} gpio_afrl_t;

// The following structure is based on register map definitions from 
// reference manual RM0091 section 8.4.10 and uses C bitfields to enable
// configuration of a single pin's alternate function 
typedef struct gpio_afrh_s {
    uint32_t afsel8 : 4;
    uint32_t afsel9 : 4;
    uint32_t afsel10 : 4;
    uint32_t afsel11 : 4;
    uint32_t afsel12 : 4;
    uint32_t afsel13 : 4;
    uint32_t afsel14 : 4;
    uint32_t afsel15 : 4;
} gpio_afrh_t;

/* 
 * The following structure is based on register map definitions from 
 * Reference Manual RM0091 section 8.4.  Be sure you understand how this
 * structure implements the register map interface to the GPIO memory-mapped 
 * peripheral - you will see (and maybe even generate) code like this all the 
 * time in embedded systems development!
 */

// Implement a GPIO* (GPIOA, GPIOB, GPIOC, ..) register interface as a structure with
// correct memory offsets from the peripheral's base address for each field in the 
// structure:
typedef volatile struct GPIO_s {
	// register name	// regsiter offset from peripheral base address & description
    gpio_mode_t MODER;  // 0x0000 Mode
    uint32_t OTYPE;     // 0x0004 Output Type
    uint32_t OSPEEDR;   // 0x0008 Output Speed
    uint32_t PUPDR;     // 0x000C Pull-up Pull-down Configuration
    uint32_t IDR;       // 0x0010 Input Data
    uint32_t ODR;       // 0x0014 Output Data
    uint32_t BSRR;      // 0x0018 Bit Set/Reset
    uint32_t LOCKR;     // 0x001C Lock (ports A, B only)
    gpio_afrl_t AFRL;   // 0x0020 Alternate Function Low
    gpio_afrh_t AFRH;   // 0x0024 Alternate Function High
    uint32_t BRR;       // 0x0028 Bit Reset
} GPIO_t;

// Create a pin datatype to enable static (compile-time) checking that a 
// function call's gpio_pin_t argument is valid.  We also assign the values
// the labels map to so they can be used directly for bit operations on 
// the IDR, ODR, BSRR, BRR registers... 
typedef enum { gpio_pin_0 = 0x0001, 
	           gpio_pin_1 = 0x0002, 
			   gpio_pin_2 = 0x0004, 
			   gpio_pin_3 = 0x0008, 
			   gpio_pin_4 = 0x0010, 
			   gpio_pin_5 = 0x0020,
			   gpio_pin_6 = 0x0040, 
			   gpio_pin_7 = 0x0080, 
			   gpio_pin_8 = 0x0100, 
			   gpio_pin_9 = 0x0200, 
			   gpio_pin_10 = 0x0400,
			   gpio_pin_11 = 0x0800, 
			   gpio_pin_12 = 0x1000, 
			   gpio_pin_13 = 0x2000, 
			   gpio_pin_14 = 0x4000, 
			   gpio_pin_15 = 0x8000 } gpio_pin_t;

// Create a pin_index type to enable static (compile-time) checking 
// that the requested pin index is valid within the range of possible pins for a GPIO port.
typedef enum { gpio_pin_index_0 = 0,
               gpio_pin_index_1 = 1,
               gpio_pin_index_2 = 2,
               gpio_pin_index_3 = 3,
               gpio_pin_index_4 = 4,
               gpio_pin_index_5 = 5,
               gpio_pin_index_6 = 6,
               gpio_pin_index_7 = 7,
               gpio_pin_index_8 = 8,
               gpio_pin_index_9 = 9,
               gpio_pin_index_10 = 10,
               gpio_pin_index_11 = 11,
               gpio_pin_index_12 = 12,
               gpio_pin_index_13 = 13,
               gpio_pin_index_14 = 14,
               gpio_pin_index_15 = 15 } gpio_pin_index_t;

// Create a pin mode type to assign to a GPIOx.moder.MODE<n> bitfield to set a pin's mode.
// Also enables static (compile-time) checking that the requested mode value is 
// within range of the GPIOx.moder.MODE bitfield.
typedef enum { gpio_mode_input = 0, 
               gpio_mode_output = 1,
               gpio_mode_alternate_function = 2,
               gpio_mode_analog = 3 } gpio_pin_mode_t;


// Getter functions:
// These read the IDR and ignore the pin mode (input, output, 
// analog, alternate-function); they may get bogus data depending
// on pin mode!  All functions return RET_GPIO_ERR_NULL_PTR if any 
// of the input arguments are unexpectedly NULL

// Get 'state' of IO pin (false = 0 = low, true = 1 = high)
ret_val_t gpio_pin_get( GPIO_t *port, gpio_pin_t pin, bool *state );

// Setter functions:
// These ignore the pin mode (input, output, analog, alternate-function); they 
// may not have any effect on pin state depending on pin mode!
// All functions return RET_GPIO_ERR_NULL_PTR if any of the 
// input arguments are unexpectedly NULL

// Use the BSSR register to set an output pin high 
ret_val_t gpio_pin_set( GPIO_t *port, gpio_pin_t pin );

// Use the BRR register to set an output pin low
ret_val_t gpio_pin_reset( GPIO_t *port, gpio_pin_t pin );

// Toggle a pin's state
ret_val_t gpio_pin_toggle( GPIO_t *port, gpio_pin_t pin );