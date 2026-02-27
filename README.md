# Quantized XOR Neural Network on STM32 Microcontroller

This repository features an optimized implementation of an XOR Neural Network developed for the **STM32F042K6 (ARM Cortex-M0)**. The project demonstrates high-efficiency embedded programming by replacing floating-point operations with **Fixed-Point Arithmetic** and transitioning from polling-based sampling to a fully **Interrupt-Driven Architecture**.

## Key Features

* **Zero Floating-Point Dependency**: Implemented using **Fixed-Point Arithmetic** ($Q3.4$ format) to maximize performance on a microcontroller without a hardware Floating Point Unit (FPU).
* **Interrupt-Driven Design**: Replaced "busy-waiting" polling with **SysTick** exceptions and **ADC conversion interrupts**, allowing for more responsive system behavior.
* **16x Performance Increase**: The quantized $NN\_qpredict()$ implementation is approximately **15.9 times faster** than the original floating-point version.
* **Memory Efficiency**: Significant reductions in both Flash memory usage (**63.6% reduction**) and RAM consumption (**45.5% reduction**).



## Technical Implementation

### Fixed-Point Quantization & Activation
Since the Cortex-M0 lacks native hardware support for floats, the neural network logic was redesigned for $int8\_t$ operations
* **Lookup Tables (LUTs)**: Activation functions like ELU and Sigmoid are handled via pre-computed quantized tables to save CPU cycles.
* **Precision Scaling**: Weights and biases were scaled by a factor of 16 ($2^4$).
* **Overflow Protection**: Used 16-bit intermediate casting during dot-product calculations to preserve accuracy before bit-shifting back to 8-bit results.

### Asynchronous Coordination
The system utilizes the **NVIC (Nested Vector Interrupt Controller)** to coordinate sampling and inference:
* **Periodic Sampling**: The **SysTick** timer triggers every 100ms to start a conversion on ADC Channel 0.
* **Chained Conversions**: The `ADC1_IRQHandler` captures the result of Channel 0 and immediately initiates a conversion for Channel 1.
* **Data Integrity**: To prevent race conditions, interrupts are globally disabled using `__asm("CPSID I")` during result scaling, then re-enabled with `__asm("CPSIE I")`.

## Benchmarks

| Metric | Floating-Point (NN_predict) | Quantized (NN_qpredict) | Improvement |
| :--- | :--- | :--- | :--- |
| **Execution Time** | 230.65 μs | 14.49 μs  | **~15.9x Faster** |
| **Flash (Text)** | 27,312 bytes  | 9,940 bytes  | **63.6% Reduction** |
| **RAM (Data+BSS)** | 880 bytes | 480 bytes  | **45.5% Reduction** |

*Note: With -O3 compiler optimization enabled, execution time drops further to **2.49 μs***

## Repository Structure

* `nn.c / nn.h`: Fixed-point Neural Network implementation and model parameters.
* `main.c`: Coordination of Systick, ADC, and USART callback functions.
* `adc.c`: Interrupt-enabled ADC driver logic.
* `gpio.c / gpio.h`: Peripheral driver for I/O pin management.
