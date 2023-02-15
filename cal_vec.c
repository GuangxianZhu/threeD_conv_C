#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float* data;
    int dim1;
    int dim2;
    int dim3;
} Tensor3D;


Tensor3D tensor3d_add(Tensor3D tensor1, Tensor3D tensor2) {
    if (tensor1.dim1 != tensor2.dim1 || tensor1.dim2 != tensor2.dim2 || tensor1.dim3 != tensor2.dim3) {
        fprintf(stderr, "Error: cannot add tensors with different dimensions\n");
        Tensor3D result = {NULL, 0, 0, 0};
        return result;
    }

    Tensor3D result;
    result.data = (float*) malloc(tensor1.dim1 * tensor1.dim2 * tensor1.dim3 * sizeof(float));
    result.dim1 = tensor1.dim1;
    result.dim2 = tensor1.dim2;
    result.dim3 = tensor1.dim3;
    for (int i = 0; i < tensor1.dim1; i++) {
        for (int j = 0; j < tensor1.dim2; j++) {
            for (int k = 0; k < tensor1.dim3; k++) {
                int index = i * tensor1.dim2 * tensor1.dim3 + j * tensor1.dim3 + k;
                result.data[index] = tensor1.data[index] + tensor2.data[index];
            }
        }
    }
    return result;
}

Tensor3D tensor3d_mul(Tensor3D t1, Tensor3D t2) {
    Tensor3D result = {0};

    int output_dim1 = t1.dim1;
    int output_dim2 = t2.dim2;
    int output_dim3 = t2.dim3;

    float* output_data = (float*) malloc(output_dim1 * output_dim2 * output_dim3 * sizeof(float));
    if (!output_data) {
        fprintf(stderr, "Error: could not allocate memory for output tensor\n");
        exit(1);
    }

    for (int i = 0; i < output_dim1 * output_dim2 * output_dim3; i++) {
        output_data[i] = 0;
    }

    Tensor3D output = {output_data, output_dim1, output_dim2, output_dim3};

    for (int i = 0; i < output_dim1; i++) {
        for (int j = 0; j < output_dim2; j++) {
            for (int k = 0; k < output_dim3; k++) {
                float sum = 0.0;
                for (int l = 0; l < t1.dim2; l++) {
                    for (int m = 0; m < t1.dim3; m++) {
                        int t1_index = i * t1.dim2 * t1.dim3 + l * t1.dim3 + m;
                        int t2_index = l * t2.dim3 * t2.dim1 + m * t2.dim1 + j * t2.dim3 + k;
                        sum += t1.data[t1_index] * t2.data[t2_index];
                    }
                }
                int output_index = i * output_dim2 * output_dim3 + j * output_dim3 + k;
                output.data[output_index] = sum;
            }
        }
    }

    result = output;

    return result;
}

Tensor3D tensor3d_scalar_mul(Tensor3D tensor, float scalar) {
    Tensor3D result;
    result.data = (float*) malloc(tensor.dim1 * tensor.dim2 * tensor.dim3 * sizeof(float));
    result.dim1 = tensor.dim1;
    result.dim2 = tensor.dim2;
    result.dim3 = tensor.dim3;

    for (int i = 0; i < tensor.dim1; i++) {
        for (int j = 0; j < tensor.dim2; j++) {
            for (int k = 0; k < tensor.dim3; k++) {
                int index = i * tensor.dim2 * tensor.dim3 + j * tensor.dim3 + k;
                result.data[index] = tensor.data[index] * scalar;
            }
        }
    }

    return result;
}

void print_tensor3d(Tensor3D tensor) {
    for (int i = 0; i < tensor.dim1; i++) {
        printf("Channel %d:\n", i);
        for (int j = 0; j < tensor.dim2; j++) {
            for (int k = 0; k < tensor.dim3; k++) {
                int index = i * tensor.dim2 * tensor.dim3 + j * tensor.dim3 + k;
                printf("%f ", tensor.data[index]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void print_tensor3d_shape(Tensor3D tensor) {
    printf("(%d, %d, %d)\n", tensor.dim1, tensor.dim2, tensor.dim3);
}

float get_tensor3d_value(Tensor3D tensor, int i, int j, int k) {
    int index = i * tensor.dim2 * tensor.dim3 + j * tensor.dim3 + k;
    return tensor.data[index];
}

Tensor3D tensor3d_negate(Tensor3D tensor) {
    Tensor3D result;
    result.data = (float*) malloc(tensor.dim1 * tensor.dim2 * tensor.dim3 * sizeof(float));
    result.dim1 = tensor.dim1;
    result.dim2 = tensor.dim2;
    result.dim3 = tensor.dim3;
    for (int i = 0; i < tensor.dim1; i++) {
        for (int j = 0; j < tensor.dim2; j++) {
            for (int k = 0; k < tensor.dim3; k++) {
                int index = i * tensor.dim2 * tensor.dim3 + j * tensor.dim3 + k;
                result.data[index] = -tensor.data[index];
            }
        }
    }
    return result;
}


Tensor3D tensor3d_conv(Tensor3D input, Tensor3D kernel, int stride, int padding) {
    int output_dim1 = (input.dim1 - kernel.dim1 + 2 * padding) / stride + 1; //output_dim1 is the number of channels
    int output_dim2 = (input.dim2 - kernel.dim2 + 2 * padding) / stride + 1; //output_dim2 is the number of rows
    int output_dim3 = (input.dim3 - kernel.dim3 + 2 * padding) / stride + 1; //output_dim3 is the number of columns

    float* output_data = (float*) malloc(output_dim1 * output_dim2 * output_dim3 * sizeof(float));
    if (!output_data) {
        fprintf(stderr, "Error: could not allocate memory for output tensor\n");
        exit(1);
    }

    for (int i = 0; i < output_dim1 * output_dim2 * output_dim3; i++) {
        output_data[i] = 0;
    }

    Tensor3D output = {output_data, output_dim1, output_dim2, output_dim3};

    for (int i = 0; i < output_dim1; i++) {
        int input_start_i = i * stride - padding;
        int input_end_i = input_start_i + kernel.dim1;

        for (int j = 0; j < output_dim2; j++) {
            int input_start_j = j * stride - padding;
            int input_end_j = input_start_j + kernel.dim2;

            for (int k = 0; k < output_dim3; k++) {
                int input_start_k = k * stride - padding;
                int input_end_k = input_start_k + kernel.dim3;

                float sum = 0.0;

                for (int l = 0; l < kernel.dim1; l++) {
                    int input_i = input_start_i + l;
                    if (input_i < 0 || input_i >= input.dim1) {
                        continue;
                    }

                    for (int m = 0; m < kernel.dim2; m++) {
                        int input_j = input_start_j + m;
                        if (input_j < 0 || input_j >= input.dim2) {
                            continue;
                        }

                        for (int n = 0; n < kernel.dim3; n++) {
                            int input_k = input_start_k + n;
                            if (input_k < 0 || input_k >= input.dim3) {
                                continue;
                            }

                            int input_index = input_i * input.dim2 * input.dim3 + input_j * input.dim3 + input_k;
                            int kernel_index = l * kernel.dim2 * kernel.dim3 + m * kernel.dim3 + n;

                            sum += input.data[input_index] * kernel.data[kernel_index];
                        }
                    }
                }

                int output_index = i * output.dim2 * output.dim3 + j * output.dim3 + k;
                output.data[output_index] = sum;
            }
        }
    }

    return output;
}

//write a function, read from a file, and return an array of floats
float* read_values_from_file(const char* filename, int* num_values) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: could not open file '%s'\n", filename);
        return NULL;
    }

    int count = 0;
    float value;
    while (fscanf(file, "%f,", &value) == 1) {
        count++;
    }

    float* values = (float*) malloc(count * sizeof(float));
    if (!values) {
        fprintf(stderr, "Error: could not allocate memory for values\n");
        fclose(file);
        return NULL;
    }

    rewind(file);

    int i = 0;
    while (fscanf(file, "%f,", &values[i]) == 1) {
        i++;
    }

    fclose(file);

    if (num_values) {
        *num_values = count;
    }

    return values;
}


int main() {

    int num_values;
    float* value_read = read_values_from_file("values.txt", &num_values);
    if (value_read) {
        printf("Read %d values:\n", num_values);
        for (int i = 0; i < num_values; i++) {
            printf("%.2f ", value_read[i]);
        }
        printf("\n");
        free(value_read);
    }

    //write tensor3d_conv
    float input_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};
    float kernel_values[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};

    // make a array for (3,16,16) tensor, value from 1 to 16*16*3
    float img_values[16*16*3];
    for (int i = 0; i < 16*16*3; i++) {
        img_values[i] = i+1;
    }
    Tensor3D img = {img_values, 3, 16, 16};
    // print_tensor3d(img);


    Tensor3D input = {input_values, 3, 3, 3};
    Tensor3D kernel = {kernel_values, 2, 2, 2};

    Tensor3D output = tensor3d_conv(img, kernel, 1, 0);
    print_tensor3d(output);
    //free(output.data);

    return 0;
}