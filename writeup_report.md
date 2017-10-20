**Behavioral Cloning**
======================

Writeup report

###  

**Behavioral Cloning Project**

The goals / steps of this project are the following:

-   Use the simulator to **collect data** of *good driving behavior*

-   **Design**, **train** and **validate** a model that *predicts a steering
    angle* from image data

-   Use the **model to drive the vehicle autonomously** around the first track
    in the simulator. The vehicle should remain on the road for an entire loop
    around the track.

    ![](writeup_images/autonomous.png)

-   Summarize the results with a written report

 

**Important points to pass the project**
========================================

#### 1. Required files submission:

My project includes the following files:

-   **model.ipynb** containing the script to create and train the model

-   **model.pdf** containing the executed script model.ipynb

-   **drive.py** for driving the car in autonomous mode

-   **model.h5** containing a trained convolution neural network

-   **writeup_report.md** summarizing the results

 

#### 2. Testing the model on the road

Using the Udacity provided simulator and my drive.py file, the car can be driven
autonomously around the track by executing

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ sh
python drive.py model.h5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

#### 3. Code readability and clearness

The **model.ipynb** file contains the code for training and saving the
convolution neural network.

The file shows the pipeline I used for training and validating the model, and it
contains comments to explain how the code works.

 

 

**Model Architecture and Training Strategy**
============================================

 

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to reuse an existing
Convolutional Neural Network, and to fine tune it.

 

My model consists of a convolution neural network **“Nvidia like”** with 3x3 but
without the last 5x5 filters

See in the section **Trained model exploration** below a visual explanation why
I decided to drop the last two layers

The model includes ELU layers to introduce nonlinearity, and the data is
normalized in the model using a Keras lambda layer (lambda x: x/127.5 - 1 ).

![](writeup_images/model%20definition.PNG)

 

#### 2. Attempts to reduce overfitting in the model

-   The model contains **dropout layers** in order to reduce overfitting (after
    the Convolutional Network , after the Flatten layer).

-   The model **was trained and validated on different data sets** to ensure
    that the model was not overfitting.

-   The model **was tested by running it through the simulator** and *ensuring
    that the vehicle could stay on the track*

 

#### 3. Dataset creation

I have recorded using the simulator a **good driving behavior.**

I have recorded different laps into different recordings ( data folder / run1,
run2, run3 ... )

Here is an example image of center lane driving:

![](writeup_images/center_2017_10_19_15_11_09_579.jpg)

I then recorded the vehicle **recovering from the left side and right side**s

of the road back to center so that the vehicle would learn

to return to the center of the lane:

 

![](writeup_images/center_2017_10_19_15_11_06_047.jpg)

![](writeup_images/center_2017_10_19_15_11_07_780.jpg)

![](writeup_images/center_2017_10_19_15_11_09_579.jpg)

 

Then I repeated this process on **track two** in order to get more data points.

 

#### 3. Data preprocessing

I put more concentration on the Data Preparation:

-   Image list extraction from csv’s

-   Adding recurrent data ( see below )

-   Splitting the sample list into Training 80% and Validation 20%

-   Resizing the images down to **128x128**

-   Image cropping to eliminate the front of the car and the upper part of the
    image ( cloud, trees... )

-   Using also the left and right images with a corrected steering angle:

    -   **left** image with steering corrected by a **+** factor ( see in the
        code ),

    -   **right** image corrected by a **-** factor ( see in the code )

    To choose the factor , I have tried many different and it seems that the
    best fit is with 0.1

    I have also tried to understand what is the best correction by recovering
    from the side and look at the steering angle ( the average is\*\* 0.03\*\*
    but it seems to me TOO small, so I have decided for **0.1** ):

    ![](writeup_images/Screen%20Shot%202017-10-08%20at%2023.55.10.png)

-   Using **Pytable**s\*\* for Big Data compatibility \*\* ( see the last
    section below )

-   Augmenting Center /Left/ Right image by flipping horizontally and reverting
    the sign of the steering .

 

 

#### 4. Recurrent Neural Network

**I have tried to introduced a recurrent pattern recognition**

In the preparation phase I have added to **the actual steering angle** multiple
previous images :

 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
samples_list_recurrent = []
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
for i,line in enumerate(samples_list):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if i% 1000 == 0 and i> 0 : print(".. recurrent data processed {}".format(i))     
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  for ix in range( max(i-1,0), max(i-5,0), -1):  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      current_steering_angle = line[3]        
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      samples_list_recurrent.append([samples_list[ix][0] ,samples_list[ix][1] ,samples_list[ix][2] ,line[3]])
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

#### 5. Training and Validation Samples exploration

Initially I had the following number of images coming from the **simulator
sampling phase:**

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reading from logfile = /ssd_data/project3/run5.csv
Reading from logfile = /ssd_data/project3/run3.csv
Reading from logfile = /ssd_data/project3/run4.csv
Reading from logfile = /ssd_data/project3/track1_run1.csv
Reading from logfile = /ssd_data/project3/run7.csv
Reading from logfile = /ssd_data/project3/run2.csv
Reading from logfile = /ssd_data/project3/run6.csv
Reading from logfile = /ssd_data/project3/run1.csv


There are 18392 samples in total 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

Adding the recurrent images:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total samples including recurrent 91946
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

And **after preprocessing and augmenting:**

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Total training samples 128x128 after augmentation and preprocessing : 441336 

Total validation samples 128x128 after augmentation and preprocessing : 110340 
... completed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

*Visual representation:*

![](writeup_images/exploring%20dataset.PNG)

 

#### 4. Training

The model used an **adam optimizer**, so the learning rate was not tuned
manually

 

Here you can see the training result:

Epoch 1/10  
13792/13791 [==============================] - 341s - loss: 0.0205 - val_loss:
0.0145  
Epoch 2/10  
13792/13791 [==============================] - 323s - loss: 0.0169 - val_loss:
0.0140  
Epoch 3/10  
13792/13791 [==============================] - 331s - loss: 0.0159 - val_loss:
0.0111  
Epoch 4/10  
13792/13791 [==============================] - 325s - loss: 0.0156 - val_loss:
0.0106  
Epoch 5/10  
13792/13791 [==============================] - 341s - loss: 0.0195 - val_loss:
0.0124  
Epoch 6/10  
13792/13791 [==============================] - 360s - loss: 0.0164 - val_loss:
0.0107  
Epoch 7/10  
13792/13791 [==============================] - 344s - loss: 0.0183 - val_loss:
0.0116  
Epoch 8/10  
13792/13791 [==============================] - 311s - loss: 0.0165 - val_loss:
0.0123  
Epoch 9/10  
13792/13791 [==============================] - 362s - loss: 0.0166 - val_loss:
0.0128  
Epoch 10/10  
13792/13791 [==============================] - 343s - loss: 0.0177 - val_loss:
0.0118

Total number of train samples: 441336 ( shape 128x128)

Batch Size : 32

Duration : 0:56:31.540967

.. model saved to model.h5

**The image below shows, that we need less than 10 epochs, the best fit is at
the epoch 4:**

 

![](writeup_images/training%20stats.PNG)

 

**I will repeat the training with a limit of 4 epochs**

 

Epoch 1/4  
13979/13978 [==============================] - 201s - loss: 0.0209 - val_loss:
0.0143  
Epoch 2/4  
13979/13978 [==============================] - 168s - loss: 0.0175 - val_loss:
0.0136  
Epoch 3/4  
13979/13978 [==============================] - 175s - loss: 0.0167 - val_loss:
0.0139  
Epoch 4/4  
13979/13978 [==============================] - 170s - loss: 0.0171 - val_loss:
0.0128  
  
Total number of train samples: 447312 ( shape 128x128)  
  
Batch Size                   : 32  
  
Duration                     : 0:11:58.673952  
  
.. model saved to model.h5

 

![](writeup_images/second training.PNG)

 

**Trained model exploration**
=============================

 

-   Loading the model from disk:

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print (" Loading drive.h5 .......")
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    from keras.models import load_model
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    from keras.models import Model
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    modelobj = load_model('model.h5')
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print (" ..... model model.h5 successfully loaded")
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

-   Loading test image:

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    image = cv2.imread('./test_images/center1.jpg')
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

-   Check if the images are cropped correctly:

![](writeup_images/keras%20cropping.PNG)

**1 - First Layer** - Conv Layer 5x5 , 24 filters ( To see all the images go
please to **model.ipynb** )

![](writeup_images/keras%20first%20layer.PNG)

**2 - Second layer** - Conv Layer 5x5 , 36 filters ( To see all the images go
please to **model.ipynb** )

![](writeup_images/keras%20second%20layer.PNG)

**3 - Third Layer** - Conv Layer 5x5 , 48 filters ( To see all the images go
please to **model.ipynb** )

![](writeup_images/keras%20layer%203.PNG)

 

**\*\* —— THERE LAYERS HAS BEEN DROPPED BECAUSE THEY ARE LOOSING INFORMATIONS
-------------\*\***

**4 - 4th** Conv Layer 3x3 , 64 filters ( To see all the images go please to
**model.ipynb** )

![](writeup_images/keras%20layer%204.PNG)

1.  **5th** Conv Layer 3x3 , 64 filters ( To see all the images go please to
    **model.ipynb** )

![](writeup_images/keras%20layer%205.PNG)

 

**Testing the model in the simulator:**
=======================================

 

The following command will load the trained **model.h5** and use the model to
make predictions on individual images in real-time and send the predicted angle
back to the server via a websocket connection.

 

It will also take a series a picture into **run1** folder that will be used
later to create a video

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   python drive.py model.h5 run1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

![](writeup_images/autonomous.png)

**​​**
====

Creating a videoclip:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python video.py run1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

And the output: **video.mp4**

 

I have also recorded a video , you can have a look on youtube ( sorry for the
quality but I had to connect remotely to my home server through Teamviewer ) :

**https://youtu.be/xrnyjzzVb08**

 

 

**Consideration about Big Data on GPU machine ( Nvidia GTX 1060 )**
===================================================================

The proposed in Udacity solution was too slow for training a lot of data.

The problem was that the generator was reading every time, every single image
from the disk, and preprocessing it again and again every time. Choosing an SSD
instead of HDD didnt helped much

 

A second solution would be to keep the preprocessed **ALL the images (64x64
pixels ) in memory,** but this solution is not scalable in a Big Data future
implementation ( because of limited RAM resources).

 

**So here instead my solution:**

#### 1. Store the preprocessed images into **Pytables**

Table definition

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
hdf5_training = open_file(ssd_folder + "/training_samples.hdf5", mode = "w", title = "Training Samples")

py_training_samples   = hdf5_training.create_earray(hdf5_training.root, \
                    'training_images', \
                    tables.UInt8Atom(), \
                    shape=( 0,resized_shape, resized_shape, 3),chunkshape=(batch_size*queue_loader_chunk ,resized_shape,resized_shape,3))
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

Image inserting ( I have done this in the **preprocess** phase )

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  for output in data_preprocess(sample_line):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        py_training_samples.append(output[0][None])
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        training_steering.append(output[1])
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

The **[None]** is used to introduce a further dimension , thats the number of
images stored.

Here you can see the Structure of **py_training_samples** :

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/training_images (EArray(381744, 128, 128, 3)) ''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  atom := UInt8Atom(shape=(), dflt=0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  maindim := 0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  flavor := 'numpy'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  byteorder := 'irrelevant'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  chunkshape := (64000, 128, 128, 3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

#### 2. Adding Multi-threades **asynchronous processing**

1 - I have designed a **Buffer** , a python **Queue** as a method to pass data
between the two Asynchronous processes

 

2 - **The first process** reads from disk and insert big chunk of data , lets
day 80000 images per time, into a Memory buffer ( a python Queue ). I use
**multiprocessing.Process** class to do it..

The great thing about Python Queue is that , if we define the **maxsize
(batch_size \* queue_loader_chunk)**, the put instruction in case the Queue is
full, will wait until will be some space free.

 

3 - **the second process** , that’s the training itself with Keras

 

#### 3. Summary

This structure now allows to maximize the different CPU cores.

 

Further, using a memory buffer it allows to optimize the data flush from SSD to
GPU.

 

As I can see from my tests, I have improved the GPU utilization from 40% to
80-90% .

![](writeup_images/improvements.png)

 

#### 4. Multi-process Queue loader

**NOTE: this solution is not implemented in the cod**e because of a unresolved
bug, see below.

Here I have tried to go further and build a multiple thread process that loads
data into the Memory.

There is still a problem , and thats about data shuffle. If you are interested
you can have a look into

**model_partitioned.ipynb** .

 

If interested in more informations, you can read my Stories on Medium:

 

[Deep Learning . Training with huge amount of data
.PART1](https://medium.com/@cristianzantedeschi/deep-learning-regression-feeding-huge-amount-of-data-to-gpu-performance-considerations-2934d32ab315)

[Deep Learning . Training with Huge amount of data .
PART2](https://medium.com/@cristianzantedeschi/github-link-https-github-com-cristianku-carnd-behavioral-cloning-project3-blob-master-model-5d27ff7394b6)

[Deep Learning . Training with Huge amount of data .
PART3](https://medium.com/@cristianzantedeschi/deep-learning-training-with-huge-amount-of-data-part3-e4bf83ff8cc6)

 

#### 5. What about shuffling the data ?

Its important to note, using this solution it is difficult to *shuffle all the
data* at each epoch change.

 

Because now I am reading data in chunk from disk, the **generator** doesnt have
all the data in hands, but only eventually this big chunk.

 

**Shuffling a Queue** is not a good choice for performance reasons.

 

So I have decided to put the **shuffle in the Queue Loader:**

 

1.  *Queue loader reads a Big Chunk from Pytable ( from disk ):*

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    chunk_batch_samples = samples[offset:offset+step]
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    chunk_batch_labels  = labels[offset:offset+step]
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2.  **SHUFFLE using sklearn.utils.shuffle**

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    chunk_batch_samples, chunk_batch_labels = shuffle(chunk_batch_samples, chunk_batch_labels )
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.  Putting data into the Python Queue (maxsize = batch_size \*
    queue_loader_chunk)

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for sample, steering in zip ( chunk_batch_samples,chunk_batch_labels):
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          samples_q.put(sample)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          labels_q.put(steering)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

#### 5. Considerations about the chunk size

The big Chunk ( the Queue size ) need **to be big enough but not too big, why
?**

1.  to increase the performance

2.  not too big ( RAM memory limitations )

3.  need to shuffle as many data as possible ( to avoid repeating training on
    same data )

Obiously the best shuffle is done on the entire dataset, but **in this Big Data
solution is not possible.**
