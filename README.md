# german-traffic-sign-recognition-multi-column-dnn
Heavily based on the 'Multi-Column Deep Neural Network for Traffic Sign Classification' paper by Ciresan et al.

It differs from the implementation in said paper in certain aspects where enough details about the methods used were not available.

- Unlike the Ciresan et al. paper, local contrast normalization is not used for preprocessing in this model.
- Instead of 25 neural nets, as used in the paper, this model implements 8 neural nets overall. These are democratically averaged to construct a multi-column deep neural net for better accuracy.

The overall accuracy achieved is 98.84%
