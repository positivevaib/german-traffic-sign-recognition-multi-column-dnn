# german-traffic-sign-recognition-multi-column-dnn
Heavily based on the 'Multi-Column Deep Neural Network for Traffic Sign Classification' paper by Ciresan et al.

It differs from the implementation in said paper in certain aspects where enough details about the methods used were not available.

Instead of 25 neural nets, as used in the paper, this model implements 10 neural nets overall. These are democratically averaged to construct a multi-column deep neural net for better accuracy.

The overall accuracy achieved is 98.84%
