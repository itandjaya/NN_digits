######### Digit image data ##########
## Data obtained from http://yann.lecun.com/exdb/mnist/
## Credits to Yann LeCun (yann at cs dot nyu dot edu) and 
## Corinna Cortes (corinna at google dot com).
## 

### IDX data format ##########
##############################

## TRAINING SET IMAGE FILE (train-images-idx3-ubyte)
##
## [offset] [type]          [value]          [description]
## 0000     32 bit integer  0x00000803(2051) magic number
## 0004     32 bit integer  60000            number of images
## 0008     32 bit integer  28               number of rows
## 0012     32 bit integer  28               number of columns
## 0016     unsigned byte   ??               pixel
## 0017     unsigned byte   ??               pixel
## ........
## xxxx     unsigned byte   ??               pixel
##
## Pixels are organized row-wise. Pixel values are 0 to 255. 
## 0 means background (white), 255 means foreground (black).

## TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
## 
## [offset] [type]          [value]          [description]
## 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
## 0004     32 bit integer  60000            number of items
## 0008     unsigned byte   ??               label
## 0009     unsigned byte   ??               label
## ........
## xxxx     unsigned byte   ??               label
## 
## The labels values are 0 to 9.


import requests
import os
import struct
import gzip
import numpy as np

from struct import unpack;

URL_train_images    =   r'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz';
URL_train_labels    =   r'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz';
URL_test_images     =   r'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz';
URL_test_labels     =   r'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz';

URLs                =   [   URL_train_images,   \
                            URL_train_labels,   \
                            URL_test_images,    \
                            URL_test_labels ];

DATA_PATH           =   os.getcwd() + r'/data/';
LABEL_MAGIC_NUMBER  =   2049;
IMAGE_MAGIC_NUMBER  =   2051;

def import_urls( URLs = URLs, redownload_data = 0):

    data = []; 

    if redownload_data:

        for url in URLs:

            r   =   requests.get(url);

            if  r.status_code   != 200:
                print(  "File download failed: ", url);
                continue;

            # Retrieve HTTP meta-data
            #print(r.status_code)
            #print(r.headers['content-type'])
            #print(r.encoding)

            file_name   =   DATA_PATH + url.split('/')[-1];

            with open(file_name, 'wb') as f:
                f.write(r.content);
    
    
    
    raw_data    = [];
    file_names  =   [];

    for (dirpath, dirnames, filenames) in os.walk(DATA_PATH):
        for filename in filenames:
            file_names.extend( [DATA_PATH + filename]);
    
    #print(file_names)

    byte_to_int     =   lambda byt: unpack("<L", byt[::-1]) [0];

    for filename in file_names:

        with gzip.open(filename, 'rb') as f:
            magic_number        =   byte_to_int(f.read(4));
            num_items           =   byte_to_int(f.read(4));
            item_size           =   1;
            data_buf            =   [];

            image_size_row  =   1;
            image_size_col  =   1;
            tag             =   r'label';


            ## Processing Image Data. Magic Number = 0x00000803(2051).
            if magic_number == IMAGE_MAGIC_NUMBER:

                image_size_row  =   byte_to_int(f.read(4));
                image_size_col  =   byte_to_int(f.read(4));

                item_size       =   image_size_row * image_size_col;
                tag             =   r'image';                
            

            buf         = f.read(   num_items * item_size);
            data        = np.frombuffer(    buf, dtype=np.uint8).astype(np.int32);

            if  tag ==  r'image':
                data    = data.reshape(     num_items,  image_size_row, image_size_col);
            
            if  tag ==  r'label':
                data    = data.reshape(     num_items,  image_size_row);

            #label       = np.frombuffer(buf, dtype=np.uint8).astype(np.int64);
            raw_data.append(    [tag, num_items, data]);

    return  raw_data;
