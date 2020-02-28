## NN_digits_main.py
## Main function to test the Neural Network - digits.
#####################################################

##  3-Layers NN: :
##  Image size is 28 x 28 pixels.
##


from import_data import import_urls;
from NN_digits import NN_digits;
from random import randint;
from numpy import save as np_save, load as np_load;

import numpy as np;
import os;
import matplotlib.pyplot as plt;    #   To plot Cost vs. # of iterations.

CURRENT_PATH        =   os.getcwd();
#CURRENT_HOME        =   (os.path.expanduser('~'))

Grad_Descent_Method, Minimize_funct_method    =   True, False;

PLOT_COST           =   False;      ##  Plot J_cost vs. # of iterations to check if J_cost converges to minimum.    
LOAD_PREV_THETAS    =   True;       ##  Load the previous trained weights. Otherwise, randomize initial weights.
CONTINUOUS_TRAINING =   False;      ##  If True, then it will train the NN (10k training iterations.)
COUNT_ERROR         =   True;       ##  Compute the error rate of the trained prediction function.
                                    ##      against the test samples (60k images). So far, ~8.5% error.


def plot_image(x):
    ##  Using matplotlib to display the gray-scaled digit image.
    ##  Input:  2D np.darray representing (28x28 pixels).

    image = np.asarray(x).squeeze();
    plt.imshow(image);
    plt.show();
    return;

def main():

    data = import_urls( redownload_data = False);   ##  Set parameter to True for initial download.
                                                    ##  Once data is present, set this to False to
                                                    ##      prevent re-downloading data.
    
    X_10k, y_10k        =  None, None;
    X_60k, y_60k        =  None, None;

    for tag, num_items, raw_data in data:
        #print(tag, num_items, raw_data.shape, np.min(raw_data), np.max(raw_data));

        if      num_items   ==  10000:
            if tag == r'image':     X_10k = raw_data;
            if tag == r'label':     y_10k = raw_data;

        if      num_items   ==  60000:
            if tag == r'image':     X_60k = raw_data;
            if tag == r'label':     y_60k = raw_data;

    digits_nn   =   NN_digits(      X_10k,              ## input data.
                                    y_10k,              ## output data.
                                    3,                  ## 3 NN layers: Input, inter-layer, output.
                                    [28*28,25,10] );    ## num of nodes for each layer.
    



    
    if Grad_Descent_Method:
        print("\nNeural Network XNOR - using GRADIENT DESCENT ITERATION\n", "#"*30, "\n");    

        # File location where learned weight is saved.
        theta_file  =   CURRENT_PATH + r'/' + 'theta.npy';

        if  LOAD_PREV_THETAS:
            flat_thetas =   np_load(    theta_file);
            digits_nn.unflatten_Thetas( flat_thetas);

            if CONTINUOUS_TRAINING:
                digits_nn.train_NN();
                np_save(    theta_file, digits_nn.flatten_Thetas());

        else:
            digits_nn.train_NN();
            np_save(    theta_file, digits_nn.flatten_Thetas());
            
            # Display final cost after learning iterations.
            print("Final Cost J = ", digits_nn.J_cost(digits_nn.a[-1]));


        if PLOT_COST:
            
            #   Plot the J Cost vs. # of iterations. J should coverge as iteration increases.
            x_axis  =   range(digits_nn.J_cost_values);
            y_axis  =   digits_nn.J_cost_values;

            plt.plot(   x_axis, y_axis, label='J_cost vs. # of Iterations');
            plt.show();
            

        # Pick 10 samples randomly from 60k samples pool.
        for i in range(0, 10):
            ith_sample  =   randint(0, 60000-1);
            x_input     =   X_60k[ith_sample].flatten();
            y_val       =   y_60k[ith_sample];
            test_res    =   digits_nn.H_funct(    x_input    );

            top_three_answers   =   "";
            for val, prob in test_res[1]:
                top_three_answers  +=  str(val) + ":" + str(prob) + "%,  ";
            top_three_answers      =   top_three_answers[:-2];

            print(  "y_val = ", y_val[0], " || Test result = ", test_res[0], "TOP THREE: ", top_three_answers);

            ## If result is not as expected, display the image.
            if not test_res[0] == y_val:
                plot_image( X_60k[ith_sample]);

        ### Calculate error rate of 60k test samples.
        ###
        if  COUNT_ERROR:

            error_cnt = 0;
            for i in range(60000):

                x_input     =   X_60k[i];
                y_val       =   y_60k[i];
                test_res    =   digits_nn.H_funct(    x_input    );

                error_cnt += (test_res[0] != y_val);

            error_rate  =   int(error_cnt/6)/100;
            print("ERROR RATE: ", error_rate, "%.");


    if Minimize_funct_method:
        print("\nNeural Network XNOR - using fmin_bfgs\n", "#"*30, "\n");

        digits_nn.init_thetas();
        digits_nn.train_NN_with_fmin();

        for i in range(0, 1000, 10):
            x_input     =   X_60k[i].flatten();
            y_val       =   y_60k[i];
            test_res    =   digits_nn.H_funct(    x_input    );
            print(  "y_val = ", y_val[0], " || Test result = ", test_res, test_res == y_val);
        
        print("\n");


    return 0;


if __name__ == "__main__":  main();
