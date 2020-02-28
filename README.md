# NN_digits - readme.txt

Neural Network - Digit image recognition.
Neural Network to classify digit images (0-9).

Files:
NN_digits_main.py:  Main function. Run as "python .\NN_digits_main.py"
NN_digits.py:       Neural Network class. Preserve input, ouptut, and values.
import_data.py:     Download and save training and test data.
theta.NPY:          Trained weight values for comparison purposes.

The network consists of 3 layers:
Layer 1: 784 input units - to interface with 28x28 pixels image.
Layer 2: 25 intermediate/hidden units.
Layer 3: 10 output units, corresponding to each digit classes (0-9).

Traning method:
Logistic regression using sigmoid activation.


DATA:

All training and test data images were obtained from MNIST database:
http://yann.lecun.com/exdb/mnist/
Credits to Yann LeCun (yann at cs dot nyu dot edu) and
Corinna Cortes (corinna at google dot com).
 

Training data consist of input and output.
Input:  Gray-scaled pixel value of images (20x20 pixels) in IDX FILE data format.
Output: Integer value 0-9 in IDX FILE data format.
See section IDX FILE FORMATS FOR THE MNIST DATABASE for detailed info.




### IDX FILE FORMATS FOR THE MNIST DATABASE


 TRAINING SET IMAGE FILE (train-images-idx3-ubyte)

 [offset] [type]          [value]          [description]
 0000     32 bit integer  0x00000803(2051) magic number
 0004     32 bit integer  60000            number of images
 0008     32 bit integer  28               number of rows
 0012     32 bit integer  28               number of columns
 0016     unsigned byte   ??               pixel
 0017     unsigned byte   ??               pixel
 ........
 xxxx     unsigned byte   ??               pixel

 Pixels are organized row-wise. Pixel values are 0 to 255. 
 0 means background (white), 255 means foreground (black).

 TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
 
 [offset] [type]          [value]          [description]
 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
 0004     32 bit integer  60000            number of items
 0008     unsigned byte   ??               label
 0009     unsigned byte   ??               label
 ........
 xxxx     unsigned byte   ??               label
 
 The labels values are 0 to 9.
