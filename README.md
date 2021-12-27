# Deep-Learning-Fruit-Recognition

DAT 565E: Fruit Image Classification

Group Number: Project Group 5 
Group Members:
Kyra Kang Yisheng Gong Haijing Zeng
 
Contents
1	Introduction	1
2	Literature Review.	1
3	Problem Description	5
4	Model Description	6
5.Data and Experimental Results	12
6.Conclusion, Discussion and Recommendations	15
7	Appendix	16
8	References	24

 
Abstract
Correctly identifying fruits not only offers customers convenience but also helps supermarkets lower the workload of cashiers and hire fewer cashiers, thus lowering the labor cost. Moreover, machines have less possibility of making mistakes especially during busy times. With the use of neural networks and transfer learning, we successfully propose methods to identify 33 common fruits in the supermarket.
The result turns out Fully- connected NN, CNN and transfer learning would be good solutions to the problem. And overall, transfer learning performs best, the accuracy is almost 100%.
1.	Introduction
Image classification technology can be widely used for detecting events for visual surveillance in the food retail industry. Supermarkets like Walmart, Schnucks, and Amazon Go Grocery can use this technology to quickly and automatically identify fruits and vegetables, which may facilitate the process of automatic payment.
Our goal is to make sure our machine can identify the correct category of the fruits, so that customers could pay the correct amount of what they bought. This means a lot to supermarket business because:
1.	It’s hard to tag every single fruit, i.e., cashiers could not scan the barcode.
2.	It offers customers more convenience when they want to check out on their own 3.It helps the supermarkets hire fewer cashiers thus lowering the cost.
4.Cashiers make mistakes especially when they are busy while machines don't.
The dataset we use is a TJ NMLO public dataset on Kaggle. The dataset contains 33 different kinds of fruits and 22495 images in total, with 16854 images in the training set and 5641 images in the test set. Due to the missing label in the test set, we decided to focus and build our models based on the training set. All the images have the same size 100x100 pixels and rgb color.
We take advantage of deep learning models to achieve our goal because neural networks and transfer learning have greatly improved machine’s ability to extract information from images, compared with traditional models. The data set we will use is Fruit Image Classification from Kaggle. And we succeeded in achieving almost 100% accuracy of identification in the end, which means it's feasible that we put it into reality.

2.	Literature Review
2.1	Dense Neural Network
Dense Neuron Network (DNN) is the simplest model used in deep learning. It has become a powerful tool of machine learning and artificial intelligence. A DNN model has a layered structure with three types: input layer, hidden layer and output layer. Each layer is made up of perceptron and each perceptron is connected with every perceptron in the next layer, which is the meaning of “dense”. There’ll be multiple hidden layers between the input and output layer. The network uses the backpropagation algorithm to optimize the weight and bias.
 
Perceptron is the smallest element in the DNN. Perceptron were developed in the 1950s and 1960s by the scientist Frank Rosenblatt, inspired by earlier work by Warren McCulloch and Walter Pitts to simulate how biological neurons work[1]. It consists of three parts: multiple input values, activation function and one output value. After performing a linear calculation of input values with weight and bias, the activation function performs some conversion to add non-linear factors to the network and it gives the final output. Some popular activation functions used in DNN are sigmoid, ReLu and softmax.
Backpropagation is an algorithm for supervised learning of artificial neural networks using gradient descent. Given an artificial neural network and an error function, the method calculates the gradient of the error function with respect to the neural network's weights. It was originally introduced in the 1970s and it became important after a famous paper by David Rumelhart, Geoffrey Hinton, and Ronald Williams in 1986.[2] It demonstrated that backpropagation made learning easier and more accurate, especially for deep learning models with multiple hidden layers.
One of the major problems DNN faced is its tendency to overfit. To overcome the problem, Steven J. Nowlan introduce R2 regularization to overcome overfit problems by adding an extra term to the error function that will penalize complexity.[3] Besides, Srivastava introduced dropout by randomly dropping units (along with their connections) from the neural network to add some noise during training[4]. Both of the two methods can help prevent overfit problems during the training process of a DNN model.

2.2	Convolutional Neural Network
Convolutional Neural Network is the widely used deep learning framework which was inspired by the visual cortex of animals. Initially it had been widely used for object recognition tasks but now it is being examined in other domains as well like object tracking , pose estimation , text detection and recognition, visual saliency detection, action recognition , scene labeling and many more .
The neocognitron in 1980 [5] is considered as the predecessor of ConvNets. LeNet was the pioneering work in Convolutional Neural Networks by LeCun et al. in 1990[6] and later improved on it [7]. It was specifically designed to classify handwritten digits and was successful in recognizing visual patterns directly from the input image without any preprocessing. But, due to lack of sufficient training data and computing power, this architecture failed to perform well in complex problems. Later in 2012, Krizhevsky et al. [8] had come up with a CNN model that succeeded in bringing down the error rate on ILSVRC competition [9]. Over the years later, their work has become one of the most influential one in the field of computer vision and used by many for trying out variations in CNN architecture. AlexNet was able to achieve remarkable results compared to the previous model of ConvNets, using purely supervised learning and without any unsupervised pre-training to keep the net simple. The architecture can be considered as a major variant of LeNet having five convolutional layers followed by three fully- connected layers. There have been various variations of AlexNet since its huge success in ILSVRC-2012 competitions. This article will serve as a guide for beginners in the area.

2.3	Transfer Learning
 
Regular machine learning and data mining techniques investigate the training data for future conclusions, with the key assumption that the future data will be in the same feature space as the training data and will have the same distribution. However, due to the scarcity of human-labeled training data, training data that remain in the same feature space or have the same distribution as future data cannot be guaranteed to be sufficient to avoid over-fitting. In real-world applications, related data from a different domain might be used in addition to data from the target domain to increase the availability of our existing knowledge about future data. Transfer learning[10] solves cross-domain learning challenges by collecting usable knowledge from data from a related domain and transferring it to a target job. With the application of transfer learning to visual categorization in recent years, some common difficulties, such as view divergence in action detection tasks and concept drifting in picture classification tasks, have been effectively overcome. Currently, transfer learning methods are widely applied to visual categorization applications such as object recognition, image classification, and human action recognition.

3.	Problem Description
There are some problems in fruits identification:
(1)	The results might be biased, because fruit datasets usually are not big enough to include all possible categories of fruits or some special cases. Initial model might not be good at recognizing a fruit with a very unique or strange appearance.
(2)	Training process may be restricted by GPU capacity. Currently both Google Colab and Kaggle offer free GPU availability to us, with a limited capacity, whether it is a usage limit or time limit. Training for computer vision usually takes hours and it would probably cause interruption and data loss if we exceed the upper limit in the middle of model training.
(3)	It's a multi-classification problem, e.g., in our dataset there are 33 categories of fruits which means it's more complex and intricate because of the high dimension.
Our methods are better than historical models because we creatively adopt transfer learning and use new network structure. Experimental results show that the discrimination degree of our transfer learning classifier outperforms other conventional classifiers. This is because:
(1)	By creatively using ResNet50V2, the performance of neural networks with additional layers has been greatly improved. However, the loss of accuracy of the test set is not stable, so we made further optimization of our model.
(2)	Then we use the MobileNet, we successfully control our loss of accuracy of the test dataset, which means the variance of the model is stable and small.
(3)	After combining them together, our model significantly improves classification training time without the loss of accuracy in multi-class classification problems.

4.	Model Description
4.1	Dense Neural Network
 
We built a full-connected Dense Neural Network model in this project. We use a sequential model with 1 flatten layer, 1 dropout layer, 3 fully-connected hidden layers and an output layer with the probability of 33 categories. Here is the summary of the model we build:

The model we built has three hidden layers, which enables it to fit the non-linear relationship between categories and fruit images. The number of hidden layers is also not very high so that it improves training and predicting performance. Besides, the number of neurons of each layer is strictly decreasing, which makes the model more stable and avoids loss of information in the training and predicting process.

4.2	Convolutional Neural Network
We built a Conv2D model for this project. We use a sequential model with 4 Conv2D layers, 4 MaxPooling2D Layers, and an output layer that returns probability of 33 categories. The summary of the model we build is as follows:
Figure 4.2 Input Layers of CNN model
 
The reason why we put 4 Conv2D layers is because after our initial try out we find our model has a serious underfitting problem(the accuracy is only 60%), so we decide to use a more complicated model, i.e., put more layers into our model. It turns out that this way works pretty well. The accuracy was improved from 60% to 98%.

4.3	Transfer learning
We experimented on 2 popular pre-trained models, ResNet50V2 and MobileNet. For the transfer learning part, we split the training data set into training part (80%) and validation part (20%). Our transfer learning model is adaptable, allowing pre-trained models to be used directly as feature extraction preprocessing or integrated into completely new models. Models trained on one problem are used as a starting point on a related problem in transfer learning. Transfer learning attempts to teach a neural network by similar means. Rather than training the neural network from scratch, it begins training with a preloaded set of weights. Usually, it will remove the topmost layers of the pretrained neural network and retrain it with new uppermost layers. The layers remaining from the previous neural network will be locked so that training does not change these weights. Only the newly added layers will be trained. Transfer learning reduces the time it takes to train a neural network model and can lead to decreased generalization error.

4.3.1	ResNet50V2
Deep residual networks have emerged as a family of extremely deep architectures showing compelling accuracy and nice convergence behaviors. We loaded a pre-trained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories. As a result, the network has learned rich feature representations for a wide range of images. The network has an image input size of 256-by-256. ResNet50V2 is a modified version of ResNet50 that performs better on the ImageNet dataset than ResNet50 and ResNet101. A change was made to the propagation formulation of the connections between blocks in ResNet50V2. On the ImageNet dataset, ResNet50V2 also performs well.
Now, we excluded the top layers of those models and added our own layers. Then we transferred their weights to our neural network. We used ensembles to gain the best predictive power. Input size is 150 by 150 pixels. We attach some part of the network here.
 
Figure 4.3.1 Input Layers of ResNet50V2
 
The invention of ResNet or residual networks, which are made up of Residual Blocks, has solved the difficulty of training deep neural networks. We notice that there is a direct connection that skips several levels in between (this may vary depending on the model). This is known as the skip connection, and it is at the heart of residual blocks. The output of the layer is no longer the same due to this skip connection. Without this skip link, the input is multiplied by the layer's weights, then a bias term is added. ResNet's skip connections alleviate the problem of disappearing gradients in deep neural networks by allowing the gradient to flow through an additional shortcut channel. These connections also aid the model by allowing it to learn the identity functions, ensuring that the higher layer performs at least as well as the lower layer, if not better.
Figure 4.3.2 Residual Block of ResNet50V2

We added one Global Average Pooling layer and two dense layers before the final output dense layer. Additional layers of the deep neural network, in the best-case scenario, can better approximate the mapping of fruit images to output classes than its shallower equivalent, reducing the error by a large margin. As a result, we anticipate ResNet50V2 to perform similarly to or better than traditional deep neural networks.
 
 

Figure 4.3.3 Output Layers of ResNet50V2

4.3.2	MobileNet
This model was designed to effectively maximize accuracy while being mindful of the restricted resources for an on-device or embedded application. MobileNets are small, low- latency, low-power models parameterized to meet the resource constraints of a variety of use cases. They can be built upon for classification, detection, embedding and segmentation similar to other popular large-scale models. Keras provides the MobileNet model. They are designed for visual applications on mobile devices and in embedded systems. MobileNets are built on a simplified design that builds lightweight deep neural networks using depth-wise separable convolutions. Two simple global hyper-parameters are introduced that efficiently trade-off latency and accuracy. These hyper-parameters enable the model builder to select the appropriate model size for their application based on the problem's constraints. Detailed research is offered on resource and accuracy trade-offs and good results on ImageNet classification compared to other popular models. The usefulness of MobileNets is then demonstrated in a variety of applications and use scenarios, including item identification, fine- grain classification, face characteristics, and large-scale geolocation.
 
 
Figure 4.3.4 Input Layers of MobileNet




Figure 4.3.5 Output Layers of MobileNet


5.	Data and Experimental Results
5.1	Data Preprocessing
Since the training dataset is well-labeled and there’s no splitation of the train data set and test data set, we need to create the new datasets for training and testing first. We use the train_test_split method to split the dataset. The dataset being large enough and we set the test_size to 0.1, which means we randomly choose 10% of the data to the test data set. The
 
splitting process is also stratified at the category level so that it ensures the balance of the proportion of different fruits between the train data set and test set. After data preprocessing, the train data set contains 15168 images and the test data set contains 1686 images.

5.2	Dense Neural Network
We compare the accuracy of the model with different dropout rates ranging from 0.05 to
0.25. The result shows that the model with 0.15 dropout rate has the best validation accuracy, which is about 0.99. We also try to use R2 regularization to help overcome the overfit problem but it doesn’t show any improvement in accuracy. The best model achieves a test accuracy of 99.06% and a test loss of 0.0354.


Figure 5.2.1 DNN training and validation accuracy

5.3	Convolutional Neural Network
Basically we tried a different architecture of CNN. In the initial tryouts we found out the CNN model has a great bias in identification, i.e., underfitting. So a more complex model would be preferred so we put more Conv2D layers and more nodes in the hidden layer(considering our so many categories). In the end we solved this problem pretty well. Our final architecture of CNN achieves a test accuracy of 99.94% and a test loss of 0.0034.

Figure 5.3.1 CNN training process
 
 
Figure 5.3.2 CNN training and validation accuracy

5.4	Transfer Learning
To evaluate the performance of transfer learning methods, we firstly look at their performance on both training and validation sets. Outside the training set and validation set, we have a test data set of 1686 fruit images. We use this test set to justify the capability of our transfer learning models. This is an out of bag evaluation and is fair to test how well the models can perform and predict the categories of unknown images.
5.4.1	ResNet50V2
The performance of neural networks with additional layers has been greatly improved by using ResNet50V2, as shown in the plots below. However, the training process is not ideally stable. In the early stage of training, a lot of turbulence can occur. ResNet50V2 eventually achieves a test accuracy of 98.5% and a test loss of 0.033.



Figure 5.4.1.1 ResNet50V2 Training and Validation Loss
 
 

Figure 5.4.1.2 ResNet50V2 Training and Validation Accuracy

5.4.2	MobileNet
The training process is more stable and validation accuracy is very close to training accuracy, indicating that MobileNet has done a good job in fitting the data. MobileNet eventually achieves a test accuracy of 100% and a test loss of 0.

Figure 5.4.2.1 MobileNet Training and Validation Loss

Figure 5.4.2.2 MobileNet Training and Validation Accuracy
 
6.	Conclusion, Discussion and Recommendations
We try different dropout rates ranging from 0.05 to o.25 in our DNN model to solve the overfitting problem. With a dropout layer, we randomly drop units (along with their connections) from the neural network. The process can add some noise during training so that it helps prevent overfitting. Based on the results, all of the DNN models with a dropout layer have better performance than the DNN model without dropout layer. The best dropout rate is
0.15 with the validation accuracy is 99%. However, a larger dropout rate such as 0.2 and 0.25 leads to a decrease in accuracy, which means that the dropout is now causing underfitting problems. Thus, there’s a trade-off for increasing dropout rage and it may cause underfit of the model. We also try R2 regularization but the validation accuracy of the model is lower than the full-connected model. We think the reason is that R2 regularization prevents the model from being too complex and causes an underfit problem.


Figure 6.1 The Validation Accuracy of Different DNN models
Transfer learning methods in our case outperform DNN and CNN. Because the vast majority of machine learning tasks are domain specific, trained models frequently fail to generalize to new situations. The real world is not like a trained data set; it contains a lot of jumbled data, and the model will make poor predictions in such circumstances. Transfer learning is defined as the ability to transfer the information of a pre-trained model to a new situation. Transfer learning has been widely employed in computer vision, owing to the abundance of excellent pre-trained models that have been trained on a vast amount of data. The application of knowledge obtained in one context to another is known as transfer learning. By using current parameters to handle "small" data problems, applying information from one model could assist minimize training time and deep learning challenges.The three methods in which transfer learning handles deep learning challenges are as follows:
1.	simpler training requirements using pre-trained data;
2.	smaller memory requirements
3.	shortened target model training
 
 

Figure 6.2 Traditional Machine Learning Compare with Transfer Learning cite: https://towardsdatascience.com/transfer-learning-in-nlp-fecc59f546e4
ResNet is concerned with computational precision. Deeper networks should not perform worse than shallower networks on paper. However, the deeper networks performed worse than the more external networks due to an optimization problem rather than overfitting. To summarize, the deeper the network, the more difficult it is to optimize. To obtain improved accuracy in computer vision, the aim is to create deeper and more sophisticated networks. On the other hand, deeper networks come at the cost of size and speed. Object detection must be performed on a computationally restricted platform in real-world applications like autonomous vehicles or robotic sights. MobileNet, a network for embedded vision applications and mobile devices, was created to address this issue.
As we can see in the test accuracies, MobileNet has given better accuracy than ResNet50V2. The ResNet-50 has accuracy 98.5% in 25 epochs and the MobileNet has accuracy 100% in 30 epochs. As we can see in the training performance of MobileNet, its accuracy is getting improved and it can be inferred that the accuracy will certainly be improved if we run the training for a greater number of epochs. However, we have shown the architecture and way to implement both the models. ResNet50V2 has more parameters to be used so it has the potential that it will show better performance as compared to the MobileNet.
We also notice that the test accuracy of all of the different models is very high. After looking over the image of our dataset we find out that the images of the same fruit are pretty similar, which means that the dataset lacks diversity. Our models may not perform very well on the images in the real world because of underfitting despite that they perform very well on the dataset. We need to update and enlarge the dataset with more real-world fruit images to train more robust models if companies want to put it into practice.
Here we build three different deep learning models to perform food image classification and transfer learning model MobileNet has the best performance, which is 100% test accuracy rate, among all of the models. We believe that fruit image classification can create great business value to the retail industry. Supermarkets can take the advantage of our models for fruit classification, which can reduce the error probability and labor cost. The customers will also save time and have a better experience with the automated process. We also notice the dataset may lack diversity and we need to enlarge the dataset with more real-world images to train a robust model before putting it into practice.
 
7.	Appendix
DNN model-full connected without dropout

DNN model-dropout rate = 0.05


DNN model-dropout rate = 0.10


DNN model-dropout rate = 0.15
 
 

DNN model-dropout rate = 0.20

DNN model-dropout rate = 0.25

DNN model-R2 regularization
 
 


Transfer learning model

Transfer learning model deployment
 
 

Training process of ResNet50V2

Training Process of MobileNet

Architecture of ResNet50V2
 


8.	References
[1]	Nielsen, M.A., 2015. Neural networks and deep learning (Vol. 25). San Francisco, CA: Determination press.
[2]	Rumelhart, D.E., Hinton, G.E. and Williams, R.J., 1986. Learning representations by back- propagating errors. nature, 323(6088), pp.533-536.
[3]	Nowlan, S.J. and Hinton, G.E., 1992. Simplifying neural networks by soft weight-sharing.
Neural computation, 4(4), pp.473-493.
[4]	Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. and Salakhutdinov, R., 2014. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1), pp.1929-1958.
[5]R Ramachandran, DC Rajeev, SG Krishnan and P Subathra, "Deep learning an overview",
IJAER, vol. 10, no. 10, pp. 25433-25448, 2015.
[6]D. H. Hubel and T. N. Wiesel, "Receptive fields and functional architecture of monkey striate cortex", The Journal of physiology, 1968.
[7]J. Fan, W. Xu, Y. Wu and Y. Gong, "Human tracking using convolutional neural networks",
Neural Networks IEEE Transactions, 2010.
[8]A. Toshev and C. Szegedy, "Deep-pose: Human pose estimation via deep neural networks",
CVPR, 2014.
[9]M. Jaderberg, A. Vedaldi and A. Zisserman, "Deep features for text spotting", ECCV, 2014.
 
[10]L. Shao, F. Zhu and X. Li, "Transfer Learning for Visual Categorization: A Survey," in IEEE Transactions on Neural Networks and Learning Systems, vol. 26, no. 5, pp. 1019-1034, May 2015, doi: 10.1109/TNNLS.2014.2330900.
![image](https://user-images.githubusercontent.com/46877714/147432941-64b27210-2586-4dea-82ae-dcc76e6fcf38.png)
