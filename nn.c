#include "nn.h"

#include <stdint.h>

#define NN_LAYER_0_NEURONS (2)
#define NN_LAYER_1_NEURONS (1)

// TODO: replace with your nn.c from the programming assignment,
// but only bring over fixed-point code!  No floating point allowed!

// Quantized NN Model Parameters
const int8_t l0_qweights[2][2] = {
	{ 43, -50, },
	{ -45, 49, },
};

const int8_t l0_qbiases[2] = { -11, -14, };

const int8_t l1_qweights[2][1] = {
	{ 56, },
	{ 55, },
};

const int8_t l1_qbiases[1] = { 4, };

// Quantized lookup table for qelu activation function
// Look up quantized result of input argument by casting the
// int8_t (qm.n) to uint8_t for the lookup table index, e.g.:
// int8_t qresult = qelu_lut[(uint8_t)qargument];

static const int8_t qelu_lut[256] = {
	0, 1, 2, 3, 4, 5, 6, 7, 
	8, 9, 10, 11, 12, 13, 14, 15, 
	16, 17, 18, 19, 20, 21, 22, 23, 
	24, 25, 26, 27, 28, 29, 30, 31, 
	32, 33, 34, 35, 36, 37, 38, 39, 
	40, 41, 42, 43, 44, 45, 46, 47, 
	48, 49, 50, 51, 52, 53, 54, 55, 
	56, 57, 58, 59, 60, 61, 62, 63, 
	64, 65, 66, 67, 68, 69, 70, 71, 
	72, 73, 74, 75, 76, 77, 78, 79, 
	80, 81, 82, 83, 84, 85, 86, 87, 
	88, 89, 90, 91, 92, 93, 94, 95, 
	96, 97, 98, 99, 100, 101, 102, 103, 
	104, 105, 106, 107, 108, 109, 110, 111, 
	112, 113, 114, 115, 116, 117, 118, 119, 
	120, 121, 122, 123, 124, 125, 126, 127, 
	-16, -16, -16, -16, -16, -16, -16, -16, 
	-16, -16, -16, -16, -16, -16, -16, -16, 
	-16, -16, -16, -16, -16, -16, -16, -16, 
	-16, -16, -16, -16, -16, -16, -16, -16, 
	-16, -16, -16, -16, -16, -16, -16, -16, 
	-16, -16, -16, -16, -16, -16, -16, -16, 
	-16, -16, -16, -16, -16, -16, -16, -16, 
	-16, -16, -16, -16, -16, -16, -16, -16, 
	-16, -16, -16, -16, -16, -16, -16, -16, 
	-16, -15, -15, -15, -15, -15, -15, -15, 
	-15, -15, -15, -15, -15, -15, -15, -15, 
	-15, -15, -15, -14, -14, -14, -14, -14, 
	-14, -14, -14, -13, -13, -13, -13, -13, 
	-12, -12, -12, -12, -11, -11, -11, -10, 
	-10, -10, -9, -9, -8, -8, -7, -7, 
	-6, -6, -5, -4, -4, -3, -2, -1, 
	};

// Quantized lookup table for qsigmoid activation function
// Look up quantized result of input argument by casting the
// int8_t (qm.n) to uint8_t for the lookup table index, e.g.:
// int8_t qresult = qsigmoid_lut[(uint8_t)qargument];

static const int8_t qsigmoid_lut[256] = {
	8, 8, 8, 9, 9, 9, 9, 10, 
	10, 10, 10, 11, 11, 11, 11, 11, 
	12, 12, 12, 12, 12, 13, 13, 13, 
	13, 13, 13, 14, 14, 14, 14, 14, 
	14, 14, 14, 14, 14, 15, 15, 15, 
	15, 15, 15, 15, 15, 15, 15, 15, 
	15, 15, 15, 15, 15, 15, 15, 16, 
	16, 16, 16, 16, 16, 16, 16, 16, 
	16, 16, 16, 16, 16, 16, 16, 16, 
	16, 16, 16, 16, 16, 16, 16, 16, 
	16, 16, 16, 16, 16, 16, 16, 16, 
	16, 16, 16, 16, 16, 16, 16, 16, 
	16, 16, 16, 16, 16, 16, 16, 16, 
	16, 16, 16, 16, 16, 16, 16, 16, 
	16, 16, 16, 16, 16, 16, 16, 16, 
	16, 16, 16, 16, 16, 16, 16, 16, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 2, 2, 2, 2, 
	2, 2, 2, 2, 2, 2, 3, 3, 
	3, 3, 3, 3, 4, 4, 4, 4, 
	4, 5, 5, 5, 5, 5, 6, 6, 
	6, 6, 7, 7, 7, 7, 8, 8, 
	};



// TODO: Implement this function using fixed-point numbers/arithmetic - no floating point!
// TODO: Comment the function as if I were sitting next to you and you are walking me through
// your implementation - what are your intentions, how do you implement them, what design 
// decisions did you think about, and then make, and why?

/* Sub-function that computes scaled weights, and bias accumulations, ensuring that type-casting
and bitshifting preserves precise fixed point representations. Added a temporary scaled array that holds
16 bit fixed point values for convenient multiplication
DESIGN DECISION: We perform the bit-shift AFTER summation to maximize precision and minimize 
rounding errors (quantization noise) before adding the bias. (converting from Q2n.2m to Q2n.m for correct addition with bias.)*/ 
void compute_neurons_q(const int8_t *inputs, int8_t *neuron_buffer) {
    neuron_buffer[0] = 0;
    neuron_buffer[1] = 0;

    int16_t scaled[2] = {0, 0}; // declaring temporary array 

    for (uint8_t i = 0; i < 2; i++) { // scaling each value using appropriate 16 bit type casting (for Q 2m.2n values) and adding them to temporary scaled array
        for (uint8_t j = 0; j < 2; j++) {
            scaled[j] += ((int16_t)inputs[i] * (int16_t)l0_qweights[i][j]);
        }
    }
    
     // DESIGN THINKING: Ensured that the currently Q2m.2n values were bit shifted to Qx.3 for bug-free addition with 
     for (uint8_t k = 0; k < 2; k++) { // adding biases and applying activation function (elu) to the sum of the scaled weights from the previous iteration.
        scaled[k] = scaled[k] >> QNN_FRACTIONAL_BITS; // converts from Qx.6 to Qx6-fractional bits (e.g. Q8.6 to Q6.3 for 3 fractional bits)
        int16_t final_val = scaled[k] + l0_qbiases[k];
        int8_t argument = (int8_t)final_val; // ensuring that the argument to the lookup table is converted to an 8 bit fixed point value.
        neuron_buffer[k] = qelu_lut[(uint8_t)argument];
    }
}


/* The NN_qpredict() function takes in a 8 bit qvalues array, calls the compute_neuron to generate
accurate 8 bit layer 0 outputs, and repeats the weight-scaling and bias accumulation for the layer 1 output
neuron, ensuring that type casting and bit shifting preserve fixed point values for convenient. I created temporaray
arrays and variables for 16 bit values to ensure that the original arrays were type-casted and that the arithmetic was smoother.
*/
int8_t NN_qpredict(const int8_t *input_qvalues) {
    int8_t hidden_neurons[2];

    // created sub-function to handle input scaling and hidden neuron computation (SEE COMMENT ^)
    compute_neurons_q(input_qvalues, hidden_neurons);

    // declaring and initilizing the output neuron as an 8 bit value.
    int8_t output[1] = {0};
     
    // declaring and initializing a temporary 16 bit fixed point variable to store multiplication and accumulation results.
    int16_t l1_scaled = (int16_t)hidden_neurons[0] * (int16_t)l1_qweights[0][0];
    l1_scaled += ((int16_t)hidden_neurons[1] * (int16_t)l1_qweights[1][0]);
    l1_scaled = l1_scaled >> QNN_FRACTIONAL_BITS; // converts from Qx.6 to Qx6-fractional bits (e.g. Q8.6 to Q6.3 for 3 fractional bits)
    l1_scaled += l1_qbiases[0]; 
    int8_t final_arg = (int8_t)l1_scaled; // ensuring that the argument into lookup table is converted to 8 bit fixed point value.
    output[0] = qsigmoid_lut[(uint8_t)final_arg]; // ensuring sigmoid function is applied to the output neuron's value.

    return output[0]; // returning the final neuron's 8 bit vaue.
}