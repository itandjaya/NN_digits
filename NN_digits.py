## NN_digits.py
## NEURAL NETWORKS: DIGIT IMAGES


import numpy as np;
from scipy.optimize import fmin_bfgs, minimize;     #   To use built-in minimize function.

class NN_digits:

    def __init__(self, X = None, y = None, num_layers = 4, arr_layer_sizes = [400,50,12,10]):

        self.var_lambda     = 0;
        self.var_alpha      = 0.5;
        self.learning_iter  = 10000;            ##  number of learning iterations.
        self.J_Cost_values  = [];

        self.NUM_LAYERS     = num_layers;       ## 4-layers NN: input layer, inter-layer1, 
                                                ##              inter-layer2, output-layer.

        ## Converting y output from size (m x 1) to size (m x 10).
        ## Value range: 0 - 9. Type = int32.
        m, n    =   np.shape(   y);
        self.y  =   np.zeros((m, 10))
        
        for i_sample in range(m):
            each_y_val  =   y[i_sample];
            self.y[i_sample, each_y_val]    =   1;

        ## Find mean and std of X data used in feature normalization.
        x_flat      =   X.flatten();
        self.mean   =   np.mean(    x_flat);
        self.std    =   np.std( x_flat);
        
        ## Converting X input from size (m x image_row x image_col) 
        ## to size (m x (image_row * image_col)).
        ## Value range: 0 - 255. Type = int32.
        m, row, col =   np.shape(   X);
        self.X      =   X.reshape(  (m, row*col) );         #   resize image data to (m x 400).
        self.X      =   self.normalize_feature(  self.X);   #   normalize gray_scale data (X-mean)/mu.
        self.X      =   np.concatenate(   ( np.ones( (m,1) ), self.X ), 1); # add bias weight 1.



        self.m, self.n      = np.shape(self.X); #   Initialize num of samples m, and num of features n.
        self.n_inter_layer  = arr_layer_sizes;  #   Initialize the num of nodes of each NN layers.

        self.Thetas = [None] * (self.NUM_LAYERS - 1);   #   List array to hold Thetas for each layer.
        self.init_thetas();                     #   Initialize random thetas (weights).

        self.z      =   [];                     #   save previous values of theta' * a.
        self.a      =   [];                     #   save each activation data of each layer.

        self.PReLU_alpha    =   0.01;           #   Constant multiplier in ReLU negative region.

        return;

    def normalize_feature(  self, X):
        return      (X - self.mean) / self.std;


    def flatten_Thetas(self):
        flattened_Thetas = np.array([]);
        for theta in self.Thetas:
            flattened_Thetas = np.concatenate(  (flattened_Thetas, theta.flatten()), 0);
        
        return flattened_Thetas;

    def unflatten_Thetas(self, flatten_Thetas):
        new_Thetas  =   [];
        i_start     =   0;

        for i_layer in range(   self.NUM_LAYERS-1):
            rows, cols          =   np.shape(self.Thetas[i_layer]);
            size                =   rows * cols;
            ft                  =   flatten_Thetas[ i_start : i_start+size];            
            theta               =   ft.reshape( rows, cols);

            new_Thetas.append(  theta  );
            i_start += size;
        
        self.Thetas =   new_Thetas;
        return new_Thetas;


    def init_thetas(self):
        ## Initialize Thetas (Weights) for each layers.
        ##  with small random numbers [-1, 1].

        np.random.seed(1);
        EPSILON         = 0.05;     # small constant multiplier.

        for i in range(len(self.Thetas)):

            this_size, next_size    = self.n_inter_layer[i:i+2];    # Theta (weight) matrix sizes for each NN layer.
               
            theta_layer     =   np.random.random(   ( next_size, this_size + 1));
            theta_layer     =   0 + EPSILON * ( (2 * theta_layer) - 1);  

            self.Thetas[i]  = theta_layer;
        
        return;
        

    def train_NN_with_fmin( self):        
        
        init_flattened_thetas   =   self.flatten_Thetas();

        #fmin_res    = fmin_bfgs(    decorated_cost, init_flattened_thetas, maxiter=400);
        fmin_res    =   fmin_bfgs(  f       = self.Cost_function_reg, \
                                    x0      = init_flattened_thetas, \
                                    fprime  = self.Theta_gradient, \
                                    maxiter = self.learning_iter,
                                    gtol    = (1e-05));
                                    #disp=True, maxiter=400, full_output = True, retall=True);
        
        #def decorated_cost(flatenned_thetas):
        #    return self.Cost_function_reg(  flatenned_thetas);

        #f_min_res  =   minimize(    fun     = decorated_cost, \
        #                            x0      = init_flattened_thetas, \
        #                            args    = (self.X, self.y), \
        #                            method  = 'TNC', \
        #                            jac     = Gradient
        #                        );

        #print(fmin_res);
        return;



    def Theta_gradient( self, flattened_thetas):

        a           =   [None] * self.NUM_LAYERS        ##  activation layers.
        z           =   [None] * self.NUM_LAYERS        ##  z = a * theta'.

        a[0]        =   self.X;
        z[0]        =   None;
        m           =   self.m;

        self.Thetas =   self.unflatten_Thetas(  flattened_thetas);

        for i_layer in range(   self.NUM_LAYERS-1):

            theta           =   self.Thetas[ i_layer];
            i_next          =   i_layer + 1;

            z[i_next]       =   np.dot( a[i_layer], theta.T);
            a[i_next]       =   np.concatenate(   ( np.ones( (m, 1)) , self.sigmoid(z[i_next]) ), 1);


        a[-1]               =   a[-1][:,1:];
        self.a, self.z      =   a, z;

        ## Back-propagation.    
        d                   =   self.a[-1] - self.y;

        new_Thetas_grad = np.array([]);

        for i_layer in range(   self.NUM_LAYERS-1):

            theta                   =   self.Thetas[-i_layer-1];
            theta_row, theta_col    =   np.shape(   theta   ); 

            theta_grad      =   (1/m)   *  np.dot(  d.T, a[-i_layer-2]   ); 

            theta_reg_term  =   (self.var_lambda/m)  *   \
                                np.concatenate( (np.zeros( (theta_row, 1)), theta[:,1:]), 1);

            theta_grad      +=  theta_reg_term;     ## adding the regulation terms.
            theta_grad      *=  self.var_alpha;     ## Use constant multiplier for faster descent.

            #self.Thetas[-i_layer-1] -= theta_grad;
            new_Thetas_grad = np.concatenate( (theta_grad.flatten(), new_Thetas_grad), 0); 

            if  i_layer < self.NUM_LAYERS-2:

                g_grad  =   self.sigmoid_gradient(  z[-i_layer-2]);
                d       =   np.dot(  d, theta[:,1:])    *   g_grad;

        return new_Thetas_grad;

    def Cost_function_reg(self, flattened_thetas):

        a           =   [None] * self.NUM_LAYERS        ##  activation layers.
        z           =   [None] * self.NUM_LAYERS        ##  z = a * theta'.

        a[0]        =   self.X;
        z[0]        =   None;
        m           =   self.m;

        self.Thetas =   self.unflatten_Thetas(  flattened_thetas);

        for i_layer in range(   self.NUM_LAYERS-1):

            theta           =   self.Thetas[ i_layer];
            i_next          =   i_layer + 1;

            z[i_next]       =   np.dot( a[i_layer], theta.T);
            a[i_next]       =   np.concatenate(   ( np.ones( (m, 1)) , self.sigmoid(z[i_next]) ), 1);


        a[-1]               =   a[-1][:,1:];
        self.a, self.z      =   a, z;

        J = self.J_cost(    a[-1])  ;       ## Compute cost for each iteration; cost should decrease
                                            ##  per iteration.
        #print("J_cost = ", J);

        return np.array([J]).flatten();

    def train_NN(self):
        
        #vect_zeros  =   np.zeros( (self.m, 1));
        #z           =   vect_zeros;

        a           =   [None] * self.NUM_LAYERS        ##  activation layers.
        z           =   [None] * self.NUM_LAYERS        ##  z = a * theta'.

        a[0]        =   self.X;
        z[0]        =   None;
        m           =   self.m;

        #######################################
        ## Feed-forward. ######################
        self.J_cost_values = [];

        for ith in range(self.learning_iter):

            ## Reduce learning rate alpha by factor of 3 for every 500 iterations.    
            if  not (ith % 2500):    self.var_alpha /= 2;

            for i_layer in range(   self.NUM_LAYERS-1):
                
                theta           =   self.Thetas[ i_layer];
                i_next          =   i_layer + 1;

                z[i_next]       =   np.dot( a[i_layer], theta.T);                
                a[i_next]       =   np.concatenate(   ( np.ones( (m, 1)) , self.sigmoid(z[i_next]) ), 1);
                #a[i_next]       =   np.concatenate(   ( np.ones( (m, 1)) , self.PReLU(z[i_next]) ), 1);

            a[-1]               =   a[-1][:,1:];
            self.a, self.zip    =   a, z;

            J   =   0;

            ## Calculate cost to make sure it converges.
            ## For learning weight parameters, Cost calculation can be skipped.
            if 1    and not (ith % 200) :
                J = self.J_cost(    a[-1])  ;       ## Compute cost for each iteration; cost should decrease
                                                    ##  per iteration.
                self.J_cost_values.append(  J);
                print(ith, " | J_cost = ", J);

                #J = self.J_cost_PReLU(    a[-1]);
                #self.J_cost_values.append(  J);
                #print(ith, " | J_cost = ", J);
            

            ## Back-propagation.  
            ########################
              
            d                   =   a[-1] - self.y;

            for i_layer in range(   self.NUM_LAYERS-1):

                theta                   =   self.Thetas[-i_layer-1];
                theta_row, theta_col    =   np.shape(   theta   );  

                theta_grad      =   (1/m)   *  np.dot(  d.T, a[-i_layer-2]   ); 

                theta_reg_term  =   (self.var_lambda/m)  *   \
                                    np.concatenate( (np.zeros( (theta_row, 1)), theta[:,1:]), 1);

                

                theta_grad      +=  theta_reg_term;     ## adding the regulation terms.
                theta_grad      *=  self.var_alpha;     ## Use constant multiplier for faster descent.

                self.Thetas[-i_layer-1] -= theta_grad;

                

                if  i_layer < self.NUM_LAYERS-2:

                    g_grad  =   self.sigmoid_gradient(  z[-i_layer-2]);
                    #g_grad  =   self.PReLU_gradient(  z[-i_layer-2]);
                    d       =   np.dot(  d, theta[:,1:])    *   g_grad; 

        return;

    def J_cost_PReLU(   self, H):
        
        square_delta    =   (H - self.y) ** 2;
        J               =   (0.5/self.m) *   np.sum( square_delta);
        return J;
        
    def J_cost(self, H):
        J = 0;

        m, n        =   self.m, self.n;
        m_y, n_y    = np.shape(    self.y);

        j = np.eye(n_y) *   (   np.dot(  self.y.T,          np.log( H)      )   \
                             +  np.dot(  ( 1.0 - self.y.T ),  np.log( 1.0 - H)  )   \
                            ); 
        
        J = (-1./m) * np.sum(    j   );

        J_reg_term = 0;
        for theta in self.Thetas:
            J_reg_term  +=   np.sum(    np.square(  theta[:,1:])    );

        J = J + (   (0.5 * self.var_lambda / m) * J_reg_term    );

        return J;


    def get_Thetas(self):
        return self.Thetas;

    def print_Thetas(self):
        for theta_ith in self.Thetas:
            print(theta_ith, "\n");
        print("\n")
        return;
    
    def sigmoid(self, Z):
        return 1. / (1. + np.exp(-Z));
    
    def sigmoid_gradient(self, Z):
        sg = self.sigmoid(Z);
        return sg * (1. - sg);
    
    def PReLU(self, Z):
        return np.maximum(-self.PReLU_alpha * Z, Z);
    
    def PReLU_gradient(self, Z):
        Z[Z<=0] =   0;
        Z[Z>0]  =   1;
        return Z;

    def H_funct(self, X):

        X   =   self.normalize_feature( X.flatten() );

        if  np.ndim(X)   ==  1:
            h   =   np.concatenate( (   np.ones((1,1)), np.array([X])    ), 1);
        
        else:
            h   =   np.concatenate( (   np.ones((1,1)), np.array(X)    ), 1);
        
        for theta in self.Thetas:
            h   =   self.sigmoid(    h.dot(theta.T));
            h   =   np.concatenate( (np.ones( (1,1)), h), 1);
        
        h       =   h[0,1:].flatten();


        idx =   np.argmax(  h );   
        ## h size: (1 x 10).
        top_three_idx   =   np.argsort(h)   [-1:-4:-1];
        top_three = [];

        for i in top_three_idx:
            top_three.append(   [i, (int(h[i]*10000))/100]);

        return idx, list(top_three);  

