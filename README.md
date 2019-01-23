# Transfer-Learning

Summary on Transfer Learning

In practice, very few people train an entire Convolutional Neural Network from scratch because it is relatively rare to have large dataset. Instead, it is common practice to pretrain a ConvNet on large dataset (such as ImageNet, which contain 1.2 million images for 1000 classes), and then use a ConvNet either as initialization or a fixed features of extractor. The three major Transfer Learning scenarios are as follows

1. ConvNet as Fixed Feature extractor:

Take a ConvNet pretrained on ImageNet, remove the last fully-connected layer (this layer’s outputs are the 1000 class scores for a different task like ImageNet), then treat the rest of the ConvNet as a fixed feature extractor for the new dataset. In an AlexNet, this would compute a 4096-D vector for every image that contains the activations of the hidden layer immediately before the classifier. We call these features CNN codes. It is important for performance that these codes are ReLUd (i.e. thresholded at zero) if they were also thresholded during the training of the ConvNet on ImageNet (as is usually the case). Once you extract the 4096-D codes for all images, train a linear classifier (e.g. Linear SVM or Softmax classifier) for the new dataset. 
