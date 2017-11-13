Project 3: Behaviorial Cloning

The Model

The model used is based on the paper from NVIDIA: End to End Learning for Self-Driving Cars. (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

The model has 5 convolutional layers and 3 fully connected layers. There are also dropout layers before the first and second fully connected layers.

Training Strategy

In order to avoid overfitting, I added 2 dropout layers.

The training date set was created from 5 laps around the track, in addition to 3 tracks going the other way. I used the left, center and right cameras and used a flipped version of each image.

Additionally I also used the shuffle function from sklearn to shuffle the data and used generators to ensure the machine didn't run out of memory while training. I also used 20% of training data for validation_samples
