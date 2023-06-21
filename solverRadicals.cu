/*
Compile and run inside a Google Colab notebook with a GPU using the commands:

!nvidia-smi
!nvcc  -o diffusion -x cu -lnvToolsExt drive/MyDrive/path/to/file/diffusionCUDARevised.cu
!./diffusion

The first line is not strictly necesary, but it lets us check what GPU we have,
probably a Tesla T4.

The second line runs the compiller. It is recomended you put this file on your
Google Drive and mount your drive to the Colab session.

The third line runs the code. Every 100 steps it will print out the current step
number.
*/

#include <string>
#include <math.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <nvToolsExt.h>
#include <stdio.h>
#include <ctime>

#define diffConst 0.1              // Diffusion constant for oxygen
#define radicalFormRate 0.00025    // Specific rate of radical formation
#define k1 0.001                   // Rate of crosslinking
#define k2 1                       // Rate of radical oxidation
#define doseRate 700               // Dose rate (Gy/hr)
#define dimX 100                   // Number of x bins, not recomened to play with this
#define dimY 100                   // Number of y bins
#define dimZ 500                   // Number of z bins
#define dimT 20000                 // Number of t bins, may need to increase
#define DSIZE (dimX * dimY * dimZ) // 5 million
#define SSIZE (dimY * dimZ * dimT) // Size of slice to save (1 billion)
#define blocks 80                  // Should be number of streaming multiprocessors x2
#define threads 1024               // Should probably be 1024

#define cudaCheckErrors(msg)                                   \
    do                                                         \
    {                                                          \
        cudaError_t __err = cudaGetLastError();                \
        if (__err != cudaSuccess)                              \
        {                                                      \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err),            \
                    __FILE__, __LINE__);                       \
            fprintf(stderr, "*** FAILED - ABORTING\n");        \
            exit(1);                                           \
        }                                                      \
    } while (0)

__global__ void finiteDiff(const float *inputVal, float *outputVal,
                           const float *inputRad, float *outputRad,
                           float *saveSlice, float *saveActivity,
                           int ds, int tStamp)
{

    for (int index = threadIdx.x + blockDim.x * blockIdx.x; index < ds; index += gridDim.x * blockDim.x)
    {

        // Determine array location
        int x = index % dimX;
        int y = ((index - x) / dimX) % dimY;
        int z = (index - x - y * dimX) / dimY / dimX;
        float radicalLoss = 0;
        float crossLinking = 0;
        float irradiationOn = 1;
        if (tStamp > 10000)
        {
            irradiationOn = 0;
        }

        // Assuming not a boundary of the array
        if (x > 0 && y > 0 && z > 0 && x < dimX - 1 && y < dimY - 1 && z < dimZ - 1)
        {
            // Oxygen concentration - initial condition
            outputVal[index] = inputVal[index];

            // Applying the laplacian for oxygen diffusion
            outputVal[index] += diffConst * (inputVal[index - 1] + inputVal[index + 1] - 2 * inputVal[index]);                     // Step the diffeq along the x dimension
            outputVal[index] += diffConst * (inputVal[index - dimX] + inputVal[index + dimX] - 2 * inputVal[index]);               // Step the diffeq along the y dimension
            outputVal[index] += diffConst * (inputVal[index - dimX * dimY] + inputVal[index + dimX * dimY] - 2 * inputVal[index]); // Step the diffeq along the z dimension

            // Radical concentration - initial condition + radical formation
            outputRad[index] = inputRad[index] + radicalFormRate * doseRate * irradiationOn;

            // Radical oxidation calculation
            // Need to account for zero concentration cases
            // Assume that the radical loss consumes fully the lowest quantity
            radicalLoss = k2 * outputRad[index] * outputVal[index];
            if (radicalLoss > outputRad[index])
            {
                if (outputVal[index] > outputRad[index])
                {
                    radicalLoss = outputRad[index];
                }
                else
                {
                    radicalLoss = outputVal[index];
                }
            }
            else if (radicalLoss > outputVal[index])
            {
                radicalLoss = outputVal[index];
            }

            // Apply radical loss
            outputRad[index] -= radicalLoss;
            outputVal[index] -= radicalLoss;

            // Crosslinking
            crossLinking = k1 * outputRad[index] * outputRad[index];
            // Boundary conditions - this quantity cannot go below zero
            if (crossLinking > outputRad[index])
            {
                outputRad[index] = 0;
            }
            else
            {
                // Normal case
                outputRad[index] -= crossLinking;
            }
        }
        else
        {
            // Fixed concentration outside
            outputVal[index] = inputVal[index];
        }

        if (x == (int)dimX / 2)
        {
            saveSlice[tStamp * dimY * dimZ + y + z * dimY] = outputVal[index];
            saveActivity[tStamp * dimY * dimZ + y + z * dimY] = radicalLoss;
        }
    }
}

// Take one step of the algorithm
void stepAlgo(float *d_inputVal, float *d_outputVal, float *d_inputRad,
              float *d_outputRad, float *d_saveSlice, float *d_saveActivity, int tStamp)
{

    // Run the main algorithm
    finiteDiff<<<blocks, threads>>>(d_inputVal, d_outputVal, d_inputRad,
                                    d_outputRad, d_saveSlice, d_saveActivity,
                                    DSIZE, tStamp);
    cudaCheckErrors("main kernel launch failure");
    cudaDeviceSynchronize();
}

void run(float *inputVal, float *inputRad, float *saveSlice, float *saveActivity)
{
    int counter = 1;

    // Declare device pointers
    float *d_inputVal;
    float *d_outputVal;
    float *d_inputRad;
    float *d_outputRad;
    float *d_saveSlice;
    float *d_saveActivity;

    // Allocate memory on the gpu
    cudaMalloc(&d_inputVal, DSIZE * sizeof(float));
    cudaMalloc(&d_outputVal, DSIZE * sizeof(float));
    cudaMalloc(&d_inputRad, DSIZE * sizeof(float));
    cudaMalloc(&d_outputRad, DSIZE * sizeof(float));
    cudaMalloc(&d_saveSlice, SSIZE * sizeof(float));
    cudaMalloc(&d_saveActivity, SSIZE * sizeof(float));
    cudaCheckErrors("cudaMalloc failure"); // error checking

    // Copy data to the GPU
    cudaMemcpy(d_inputVal, inputVal, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputVal, inputVal, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputRad, inputRad, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputRad, inputRad, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    std::time_t msStart = std::time(nullptr);
    while (counter <= dimT)
    {
        // Print out the counter every 1000 iterations
        if ((counter - 1) % 1000 == 0)
        {
            std::cout << counter - 1 << std::endl;
        }
        // Run 1 step of the algorithm
        stepAlgo(d_inputVal, d_outputVal, d_inputRad, d_outputRad, d_saveSlice, d_saveActivity, counter - 1);
        counter++;

        // Run another step but with the input and output arrays flipped so
        // the memory doesn't need copied
        stepAlgo(d_outputVal, d_inputVal, d_outputRad, d_inputRad, d_saveSlice, d_saveActivity, counter - 1);
        counter++;
    }
    std::time_t msEnd = std::time(nullptr);

    // Give timing information
    std::cout << double(msEnd - msStart) * double(1000) / double(counter) << " ms per step\n";

    // Copy data off the GPU
    cudaMemcpy(inputVal, d_inputVal, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(saveSlice, d_saveSlice, SSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(saveActivity, d_saveActivity, SSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

    // Free the memory on the GPU
    cudaFree(d_inputVal);
    cudaFree(d_outputVal);
    cudaFree(d_inputRad);
    cudaFree(d_outputRad);
    cudaFree(d_saveSlice);
    cudaFree(d_saveActivity);
}

int main(int argc, char *argv[])
{

    // Allocate arrays for data storage
    float *inputArray = new float[DSIZE];
    float *radArray = new float[DSIZE];
    float *saveSlice = new float[SSIZE];
    float *saveActivity = new float[SSIZE];

    float inside = 1;  // Concentration inside
    float outside = 1; // Concentration outside
    float radVal = 0;  // Initial radical concentration

    // Initialize the first array
    for (int x = 0; x < dimX; x++)
    {
        for (int y = 0; y < dimY; y++)
        {
            for (int z = 0; z < dimZ; z++)
            {
                int index = x + dimX * (y + z * dimY); // Location in flat array
                if (x > 0 && y > 0 && z > 0 && x < dimX - 1 && y < dimY - 1 && z < dimZ - 1)
                {
                    // inside
                    inputArray[index] = inside;
                }
                else
                {
                    // outside
                    inputArray[index] = outside;
                }
                radArray[index] = radVal;
            }
        }
    }

    // Run the algorithm
    run(inputArray, radArray, saveSlice, saveActivity);

    // Store the data in a binary file
    // This can be opened in python with:
    // np.fromfile("data.dat", dtype=np.float32)
    // data = np.reshape(data,(10000,100,100))
    // Shape is (t, y, x)
    FILE *data = fopen("data.dat", "wb");
    fwrite(saveSlice, sizeof(float), SSIZE, data);
    fclose(data);

    FILE *activity = fopen("activity.dat", "wb");
    fwrite(saveActivity, sizeof(float), SSIZE, activity);
    fclose(activity);

    // Free memory
    free(inputArray);
    free(radArray);
    free(saveSlice);
    free(saveActivity);

    return (0);
}
