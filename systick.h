#pragma once

#include <stdint.h>

typedef volatile struct SYSTICK_s {
    uint32_t CSR;
    uint32_t RVR;
    uint32_t CVR;
    uint32_t CALIB;
} systick_t;

#define SYSTICK_CSR_CLKSOURCE (4)
#define SYSTICK_CSR_TICKINT   (2)
#define SYSTICK_CSR_ENABLE    (1)

#define SYSTICK ((systick_t *)0xE000E010)

void systick_init( void(*cbfn)(void) );