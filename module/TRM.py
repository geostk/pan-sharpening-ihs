import pycuda.driver as cuda
import pycuda.tools
import pycuda.autoinit
import numpy as np
import numpy.linalg as la
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import example




class TransformationManager:
    def __init__(self,
                 normalized_red,
                 normalized_green,
                 normalized_blue,
                 panchrom_in_ihs):
        self.normalized_red = normalized_red
        self.normalized_green = normalized_green
        self.normalized_blue = normalized_blue
        self.normalized_panchromatic = panchrom_in_ihs

    def arithmetic_average(self):
        red_out = 0.5 * (self.normalized_red + self.normalized_panchromatic)
        green_out = 0.5 * (self.normalized_green + self.normalized_panchromatic)
        blue_out = 0.5 * (self.normalized_blue + self.normalized_panchromatic)

        return red_out, green_out, blue_out

    def arithmetic_average_gpu(self):

        def cuda_operations(input_colour, input_pan):
            # mod = SourceModule("""
            # __global__ void multiply_them(float *dest, float *input_colour, float *input_pan)
            # {
            #     int n_x = blockDim.x*gridDim.x;
            #     int i = threadIdx.x + blockDim.x*blockIdx.x;
            #     int j = threadIdx.y + blockDim.y*blockIdx.y;
            #     int threadId = j*n_x+i;
            #     dest[threadId] = 0.5 * (input_colour[threadId] + input_pan[threadId]);
            # }
            # """)
            #
            # multiply_them = mod.get_function("multiply_them")
            #
            # input_colour = input_colour.astype(np.float32)
            # input_pan = input_pan.astype(np.float32)
            #
            # dest = np.zeros_like(input_colour)
            # multiply_them(
            #     cuda.Out(dest), cuda.In(input_colour), cuda.In(input_pan),
            #     block=(16, 16, 1), grid=(16, 16))

            a_gpu = gpuarray.to_gpu(input_colour.astype(np.float32))
            b_gpu = gpuarray.to_gpu(input_pan.astype(np.float32))
            dest = (0.5 * (a_gpu + b_gpu)).get()
            return dest

        red_out = cuda_operations(self.normalized_red, self.normalized_panchromatic)
        green_out = cuda_operations(self.normalized_green, self.normalized_panchromatic)
        blue_out = cuda_operations(self.normalized_blue, self.normalized_panchromatic)

        return red_out, green_out, blue_out

    def rgb_to_ihs_cpu(self):
        red_in = self.normalized_red / 255
        green_in = self.normalized_green / 255
        blue_in = self.normalized_blue / 255

        # intensity = np.empty([self.normalized_red.shape[0], self.normalized_red.shape[1]], dtype=np.float32)
        hue = np.empty([self.normalized_red.shape[0], self.normalized_red.shape[1]], dtype=np.float32)
        saturation = np.zeros([self.normalized_red.shape[0], self.normalized_red.shape[1]], dtype=np.float32)

        intensity = red_in + green_in + blue_in

        for x in range(red_in.shape[0]):
            for y in range(red_in.shape[1]):
                values = ([red_in[x][y], green_in[x][y], blue_in[x][y]])

                # intensity & saturation

                if values[0] == values[1] == values[2]:
                    if 0.0 > values[0] <= 0.33:
                        hue[x][y] = 0
                    elif 0.33 > values[0] <= 0.66:
                        hue[x][y] = 1.3
                    else:
                        hue[x][y] = 2.6

                # Blau
                elif values.index(min(values)) == 2:
                    div = intensity[x][y] - (3 * blue_in[x][y])

                    if div == 0:
                        hue[x][y] = 0
                    else:
                        hue[x][y] = (green_in[x][y] - blue_in[x][y]) / div

                    if intensity[x][y] > 0:
                        saturation[x][y] = div / intensity[x][y]

                # Rot
                elif values.index(min(values)) == 0:
                    div = intensity[x][y] - 3 * red_in[x][y]
                    hue[x][y] = ((blue_in[x][y] - red_in[x][y]) / div) + 1

                    if intensity[x][y] > 0:
                        saturation[x][y] = div / intensity[x][y]

                # Grün
                elif values.index(min(values)) == 1:
                    div = (intensity[x][y] - 3 * green_in[x][y])
                    hue[x][y] = ((red_in[x][y] - green_in[x][y]) / div) + 2

                    if intensity[x][y] > 0:
                        saturation[x][y] = div / intensity[x][y]

                else:
                    print("Error: During calculating hue on position" + str([x] + [y]))

                if np.isnan(hue[x][y]):
                    hue[x][y] = 1
                    print("Nan value detectet in hue " + str([x] + [y]) + " Set it to 1")
                    print(values)

                if hue[x][y] > 3.0:
                    print("Hue above 3.0, that may cause some trouble...")

                if saturation[x][y] > 1.0:
                    print("Saturation above 1.0, that may cause some trouble...")

        return intensity, hue, saturation

    def intensity_mixer_cpu(self, intensity, old, new):
        return float(old) * intensity + float(new) * self.normalized_panchromatic

    def ihs_to_rgb_cpu(self, intensity, hue, saturation):

        red = np.empty([self.normalized_red.shape[0], self.normalized_red.shape[1]], dtype=np.float32)
        green = np.empty([self.normalized_red.shape[0], self.normalized_red.shape[1]], dtype=np.float32)
        blue = np.empty([self.normalized_red.shape[0], self.normalized_red.shape[1]], dtype=np.float32)

        for x in range(intensity.shape[0]):
            for y in range(intensity.shape[1]):

                if 0 <= hue[x][y] <= 1:
                    red[x][y] = intensity[x][y] * (1 + 2 * saturation[x][y] - 3 * saturation[x][y] * hue[x][y]) / 3
                    green[x][y] = intensity[x][y] * (1 - saturation[x][y] + 3 * saturation[x][y] * hue[x][y]) / 3
                    blue[x][y] = intensity[x][y] * (1 - saturation[x][y]) / 3

                elif 1 <= hue[x][y] <= 2:
                    red[x][y] = intensity[x][y] * (1 - saturation[x][y]) / 3
                    green[x][y] = intensity[x][y] * (
                    1 + 2 * saturation[x][y] - 3 * saturation[x][y] * (hue[x][y] - 1)) / 3
                    blue[x][y] = intensity[x][y] * (1 - saturation[x][y] + 3 * saturation[x][y] * (hue[x][y] - 1)) / 3

                elif 2 <= hue[x][y] <= 3:
                    red[x][y] = intensity[x][y] * (1 - saturation[x][y] + 3 * saturation[x][y] * (hue[x][y] - 2)) / 3
                    green[x][y] = intensity[x][y] * (1 - saturation[x][y]) / 3
                    blue[x][y] = intensity[x][y] * (
                    1 + 2 * saturation[x][y] - 3 * saturation[x][y] * (hue[x][y] - 2)) / 3
                else:
                    pass
                    # print("Error while transforming back to RGB")

        return 255 * red, 255 * green, 255 * blue

    def rgb_to_ihs_gpu(self):

        red_in = self.normalized_red / 255
        green_in = self.normalized_green / 255
        blue_in = self.normalized_blue / 255

        mod = SourceModule("""
#include <stdio.h>

__global__ void multiply_them(float *hue, float *saturation, float *input_red,
		float *input_green, float *input_blue, float *intensity) {

	int n_x = blockDim.x * gridDim.x;
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int threadId = j * n_x + i;

	if (input_red[threadId] == input_green[threadId] == input_blue[threadId]) {
		if (input_red[threadId] >= 0.0 && input_red[threadId] <= 0.33) {
			hue[threadId] = 0.0;
			//printf("all values equal, hue set to 0.0 \\n");
		}

		else if (input_red[threadId] > 0.33 && input_red[threadId] <= 0.66) {
			hue[threadId] = 1.3;
			//printf("all values equal, hue set to 1.3 \\n");
		}

		else {
			hue[threadId] = 2.3;
			//printf("all values equal, hue set to 2.6 \\n");
		}

	}
	//Blau am größten
	else if (input_blue[threadId] > input_red[threadId]
			&& input_blue[threadId] > input_green[threadId]) {
		float div_blue = intensity[threadId] - (3 * input_blue[threadId]);

		if (div_blue == 0) {
			hue[threadId] = 0;
		} else {
			hue[threadId] = (input_green[threadId] - input_blue[threadId])
					/ div_blue;
		}

		if (intensity[threadId] > 0) {
			saturation[threadId] = div_blue / intensity[threadId];
		} else {
			saturation[threadId] = 0;
		}

	}

	//Rot            
	else if (input_red[threadId] > input_blue[threadId]
			&& input_red[threadId] > input_green[threadId]) {
		float div_red = intensity[threadId] - 3 * input_red[threadId];
		hue[threadId] = ((input_blue[threadId] - input_red[threadId]) / div_red)
				+ 1;

		if (intensity[threadId] > 0) {
			saturation[threadId] = div_red / intensity[threadId];
		}

		else {
			saturation[threadId] = 0;
		}
	}

	//Grün

	else if (input_green[threadId] > input_blue[threadId]
			&& input_green[threadId] > input_red[threadId]) {
		float div_green = intensity[threadId] - 3 * input_green[threadId];
		hue[threadId] = ((input_red[threadId] - input_green[threadId])
				/ div_green) + 2;

		if (intensity[threadId] > 0) {
			saturation[threadId] = div_green / intensity[threadId];
		}

		else {
			saturation[threadId] = 0;
		}
	}

	else {
	printf("no color \\n");
	
	float div_no_color = intensity[threadId] - 3 * input_green[threadId];
		hue[threadId] = ((input_red[threadId] - input_green[threadId]) / div_no_color) + 2;

		if (intensity[threadId] > 0) {
			saturation[threadId] = div_no_color / intensity[threadId];
		}

		else {
			saturation[threadId] = 0;
		}
}
}
        """)

        multiply_them = mod.get_function("multiply_them")

        input_red = red_in.astype(np.float32)
        input_green = green_in.astype(np.float32)
        input_blue = blue_in.astype(np.float32)
        intensity = (input_red + input_green + input_blue) / 3
        hue = np.zeros_like(input_red)
        saturation = np.zeros_like(input_red)

        multiply_them(
            cuda.Out(hue),
            cuda.Out(saturation),
            cuda.In(input_red),
            cuda.In(input_green),
            cuda.In(input_blue),
            cuda.In(intensity),
            block=(4, 4, 1), grid=(53, 50)
        )

        return intensity, hue, saturation

    def intensity_mixer_gpu(self, intensity, old, new):
        return float(old) * intensity + float(new) * self.normalized_panchromatic / 255

    def ihs_to_rgb_gpu(self, intensity, hue, saturation):

        red = np.empty([self.normalized_red.shape[0], self.normalized_red.shape[1]], dtype=np.float32)
        green = np.empty([self.normalized_red.shape[0], self.normalized_red.shape[1]], dtype=np.float32)
        blue = np.empty([self.normalized_red.shape[0], self.normalized_red.shape[1]], dtype=np.float32)

        mod = SourceModule("""
        #include <stdio.h>

        __global__ void multiply_them(float *red, float *green, float *blue, float *intensity, float *hue, float *saturation)
        {

            int n_x = blockDim.x*gridDim.x;
            int i = threadIdx.x + blockDim.x*blockIdx.x;
            int j = threadIdx.y + blockDim.y*blockIdx.y;
            int threadId = j*n_x+i;
            
            
            if (hue[threadId] >= 0 && hue[threadId] <= 1) 
            {
                red[threadId] = intensity[threadId] * (1 + 2 * saturation[threadId] - 3 * saturation[threadId] * hue[threadId]) / 3;
                green[threadId] = intensity[threadId] * (1 - saturation[threadId] + 3 * saturation[threadId] * hue[threadId]) / 3;
                blue[threadId] = intensity[threadId] * (1 - saturation[threadId]) / 3;
            }
    
            else if (hue[threadId] >= 1 && hue[threadId] <= 2)
            {
                red[threadId] = intensity[threadId] * (1 - saturation[threadId]) / 3;
                green[threadId] = intensity[threadId] * (1 + 2 * saturation[threadId] - 3 * saturation[threadId] * (hue[threadId] - 1)) / 3;
                blue[threadId] = intensity[threadId] * (1 - saturation[threadId] + 3 * saturation[threadId] * (hue[threadId] - 1)) / 3;
            }
    
            else if (hue[threadId] >= 2 && hue[threadId] <= 3)
            {
                red[threadId] = intensity[threadId] * (1 - saturation[threadId] + 3 * saturation[threadId] * (hue[threadId] - 2)) / 3;
                green[threadId] = intensity[threadId] * (1 - saturation[threadId])/3;
                blue[threadId] = intensity[threadId] * (1 + 2 * saturation[threadId] - 3 * saturation[threadId] * (hue[threadId] - 2))/3;
            }
            
            else
            {
                printf("something went wrong \\n");
                red[threadId] = 0;
                green[threadId] = 0;
                blue[threadId] = 0;
            }
                      
        }
        """)

        multiply_them = mod.get_function("multiply_them")

        multiply_them(
            cuda.Out(red), cuda.Out(green), cuda.Out(blue), cuda.In(intensity), cuda.In(hue), cuda.In(saturation),
            block=(4, 4, 1), grid=(53, 50))
        # print("red: \n")
        print(hue[0][0])
        print(intensity[0][0])
        # print("Sat from ihs to rgb")
        print(saturation[0][0])
        print("---")
        print(red[0][0] * 255)
        # print("---------------------------")
        # print("green: \n")
        # print(green)
        # print("---------------------------")
        # print("sat: \n")
        # print(blue)
        # print("---------------------------")
        return red * 255, green * 255, blue * 255

    def matrix(self):

        red = self.normalized_red
        green = self.normalized_green
        blue = self.normalized_blue
        pan = self.normalized_panchromatic

        verrechnugsmatrix = np.matrix([[1 / 3, 1 / 3, 1 / 3],
                                       [-1 * (np.sqrt(2) / 6), -1 * (np.sqrt(2) / 6), (2 * np.sqrt(2) / 6)],
                                       [1 / np.sqrt(2), -1 / np.sqrt(2), 0]])

        rueckrechnungsmatrix = np.matrix([[1, -1 / np.sqrt(2), 1 / np.sqrt(2)],
                                          [1, -1 / np.sqrt(2), -1 / np.sqrt(2)],
                                          [1, np.sqrt(2), 0]])

        r1 = np.empty([red.shape[0], red.shape[1]], dtype=np.float32)
        g1 = np.empty([red.shape[0], red.shape[1]], dtype=np.float32)
        b1 = np.empty([red.shape[0], red.shape[1]], dtype=np.float32)

        # rgb_matrix = np.matrix([[red], [green], [blue]])
        for x in range(red.shape[0]):
            for y in range(red.shape[1]):
                ergebnis_matrix = verrechnugsmatrix * np.matrix([[red[x][y]],
                                                                 [green[x][y]],
                                                                 [blue[x][y]]])

                output_matrix = rueckrechnungsmatrix * np.matrix([[pan[x][y]],
                                                                  [ergebnis_matrix.item(1)],
                                                                  [ergebnis_matrix.item(2)]])

                r1[x][y] = output_matrix.item(0)
                g1[x][y] = output_matrix.item(1)
                b1[x][y] = output_matrix.item(2)


        return r1, g1, b1

    def matrix_gpu(self):

        red = self.normalized_red
        green = self.normalized_green
        blue = self.normalized_blue
        pan = self.normalized_panchromatic

        mod = SourceModule("""
        #include <stdio.h>
        #include <math.h>

        __global__ void multiply_them(
        float *r1, 
        float *g1, 
        float *b1,
        float *input_red, 
        float *input_green, 
        float *input_blue, 
        float *input_pan,
        int *input_xy        
        ) {
            
            int n_x = blockDim.x * gridDim.x;
            int i = threadIdx.x + blockDim.x * blockIdx.x;
            int j = threadIdx.y + blockDim.y * blockIdx.y;
            int threadId = j * n_x + i;

            int n_iter  = 2048;
        	
            float matrix[3][3] = 
            {{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0},
            {-1.0 * (sqrt(2.0) / 6.0), -1.0 * (sqrt(2.0) / 6.0), (2.0 * sqrt(2.0) / 6.0)},
            {1.0 / sqrt(2.0), -1.0 / sqrt(2.0), 0.0}};
            
            float matrix_back[3][3] = 
            {{1.0, -1.0 / sqrt(2.0), 1.0 / sqrt(2.0)},
            {1.0, -1.0 / sqrt(2.0), -1.0 / sqrt(2.0)},
            {1.0, sqrt(2.0), 0.0}};
            
        
            float zm[3][1];           
            int x, y;

            for ( x = 0; x < n_iter; x++ ) { 
                for ( y = 0; y < n_iter; y++ ) { 
                    
                zm[0][0] = matrix[0][0] * input_red[threadId] + matrix[1][0] * input_green[threadId] + matrix[2][0] * input_blue[threadId];            
                zm[0][1] = matrix[0][1] * input_red[threadId] + matrix[1][1] * input_green[threadId] + matrix[2][1] * input_blue[threadId];           
                zm[0][2] = matrix[0][2] * input_red[threadId] + matrix[1][2] * input_green[threadId] + matrix[2][2] * input_blue[threadId];

                
                r1[threadId] = matrix_back[0][0] * input_pan[threadId] + matrix_back[1][0] * zm[0][1] + matrix_back[2][0] * zm[0][2];
                g1[threadId] = matrix_back[0][1] * input_pan[threadId] + matrix_back[1][1] * zm[0][1] + matrix_back[2][1] * zm[0][2];
                b1[threadId] = matrix_back[0][2] * input_pan[threadId] + matrix_back[1][2] * zm[0][1] + matrix_back[2][2] * zm[0][2];
                

            }
            }
                        
        }
        
        """)


        multiply_them = mod.get_function("multiply_them")


        # Array größe X Achse (Y die selbe)
        n_iter = ([[red.shape[0]], [red.shape[1]]])

        # Leere Arrays auf die GPU schieben
        r1 = gpuarray.empty((red.shape[0], red.shape[0]), np.float32)
        g1 = gpuarray.empty((red.shape[0], red.shape[0]), np.float32)
        b1 = gpuarray.empty((red.shape[0], red.shape[0]), np.float32)



        # Arrays mit alten Werten auf die GPU schieben
        input_red = gpuarray.to_gpu(red.astype(np.float32))
        input_green = gpuarray.to_gpu(green.astype(np.float32))
        input_blue = gpuarray.to_gpu(blue.astype(np.float32))
        input_pan = gpuarray.to_gpu(pan.astype(np.float32))

        multiply_them(
            r1,
            g1,
            b1,
            input_red,
            input_green,
            input_blue,
            input_pan,
            block=(4, 4, 1)
        )

        return r1.get(), g1.get(), b1.get()

    def matrix_cpu_c_module(self):

        red = self.normalized_red
        green = self.normalized_green
        blue = self.normalized_blue
        pan = self.normalized_panchromatic
        r1, g1, b1 = \
            example.multiply_them(
                self.normalized_red.shape[0],
                self.normalized_red.shape[1],
                red.flatten(),
                green.flatten(),
                blue.flatten(),
                pan.flatten())

        #r1, g1, b1 = example.multiply_them(self.normalized_red.shape[0], self.normalized_red.shape[1], red, green, blue, pan)

        return r1, g1, b1