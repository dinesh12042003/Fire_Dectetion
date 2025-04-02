# Fire Dectetion 

1	INTRODUCTION

1.1	Problem definition
         Forest fires present one of the main causes of environmental hazards that have many negative results in different aspects of life. At present, more than 220,000 forest fires occur per year in the world, and more than  6.4 million  hm2 of forests are affected,  accounting for more than 0.23%  of the world's forests.  Forest fire emits large amounts of greenhouse gases, which aggravate atmospheric and environmental pollution. 
       In addition, serious forest fires will also lead to human and property loss. In the year 2015, there were 3,703 forest fires that broke out throughout China, including one particularly serious fire, resulting in approximately 400 million-yuan losses in total. 
       Forest fire prediction has been in research for nearly a hundred years. Early forest fire monitoring relied on manual monitoring, effectively, though, with certain delays the safety of personnel cannot be guaranteed. Infrared images are utilised to monitor open flames, however, as the smoke of early fire  is much more obvious compared to open flames, therefore, it is unable to obtain fire information in time, resulting in losses of the best opportunity to put out the fire.  
       Using satellites and drones to monitor fire, and achieving a remarkable result, costs so much, consequently, leads to its inability to be deployed in a large scale. Moreover, video surveillance obtains one-sided information and may not monitor the forest accurately. With the impact of human activities and changes in global climate, forest fires have been intensified for the past 10 years. Finding an effective way to predict forest  fire hazard  has become  top priority in the forest fire research field.

1.2	Objective of the project
       In recent years, the development of electrical equipment and new organic materials has brought about huge fire hazards, resulting in huge casualties and economic losses worldwide.
       In order to realize the early warning of fire, the characteristics of temperature, smoke concentration and carbon monoxide sensor data in the initial stage of fire were analyzed in this study, and a back propagation neural network was chosen to achieve the fusion of the three kinds of fire data. In addition, this study adopted the methods of non-uniform sampling and trend extraction to improve the performance of fire warning algorithm in the early stage of fire. 
       The fire warning algorithm proposed in this project is designed for indoor/outdoor fire scenarios. To build robust and reliable fire warning systems, multi-sensor systems need to be exposed to more types of fires and nuisances. Hence, the next step is to expand the fire dataset for the multi-sensor fusion model so that is able to cover flammable materials commonly found in home. 
       Moreover, it is also our goal to build a wireless sensor network (WSN) to test the algorithm in a realistic fire scenario and apply it to the Internet of Things (IoT).


1.3	Limitations of the Project
       Due to decreased false positive and false negative rates, there is a little possibility that our fire detection devices, which apply various algorithms to detect fire, will occasionally issue false warnings.
Our project is among the finest at detecting fire.




2	ANALYSIS

2.1	Introduction
Fire Parameters
In the early stage of fires, the phenomena and products produced by combustion of different materials are different, but there are also some similarities, like the release of heat and production of smoke. These common burning products (temperature, humidity, smoke, CO2, CO , etc.) are called fire characteristic parameters, also known as fire parameters. The fire early warning technology based on multi-sensor data fusion relies on the detection and processing of fire parameters. Thus, selecting suitable fire parameters is important for fire early warning algorithms. The following are the fire parameters used in this study:
(1)	Temperature
When a fire occurs, heat is released and the temperature of surrounding environment increases. Temperature is the earliest and most versatile indicator in fire warning technology. However, the alarm threshold of temperature is usually greater than 60 °C to reduce the false alarm caused by weather changes. As a result, temperature is rarely used as a lone indicator. Nevertheless, it is a good balance metric for fire early warning algorithms based on multi-sensor data fusion model.
(2)	Smoke concentration
Smoke is an important feature in the characterization of fire . Smoke concentration is one of the most obvious indicators of fires. The current market share of smoke detectors is around 60% [15], though consumers are also affected by the failure or false alarms of smoke detectors. These problems can be effectively overcome when smoke is used as one of the indictors in a multi-sensor data fusion model.
(3)	Carbon monoxide
Normally, the content of carbon monoxide (CO) in the air is low, and it increases rapidly in the event of a fire . There are few events that generate enough CO to trigger a fire detector, unlike carbon dioxide (CO2). Therefore, carbon monoxide is an effective indicator for fire detection.
Although humidity and carbon dioxide are also common byproducts of combustion, they are closely related to the environment. Hence, temperature, smoke concentration and carbon monoxide were selected as indicators of fire detection in this study.
(4)	Trend values of fire parameters
Usually, fire parameters fluctuate randomly and irregularly due to disturbance in the normal environment. When a fire occurs, fire parameters have obvious and continuous positive or negative trend characteristics . Hence, the trend values of fire parameters also can be used as effective indicators to distinguish fire signals from environmental interference.

2.2	Software requirement specification
2.2.1	Software Requirement
•	Python Interpreter
•	Python IDE 
2.2.2	Hardware Requirement
•	Min-Intel i5 10th Gen/AMD Athlon 300U processor
•	8 GB Ram
•	CPU – 2.5 GHz




2.3	Existing System
Image-Based Fire Detection
       A novel approach for forest fire detection using the image processing technique. A rule-based colour model for fire pixel classification is used. The algorithms uses RGB and YC b C r colour space. The advantage of using YC b C r colour space is that it can separate the luminance from the chrominance more effectively than RGB colour space. The performance of the algorithm is tested on two sets of images, one of which contains fire; the other contains fire-like regions. Standard methods are used for calculating the performance of the algorithm. The proposed method has both a higher detection rate and a lower false alarm rate. Since the algorithm is cheap in computation, it can be used for real-time forest fire detection.
Sensors-Based Fire Detection 
Perceptron
      A perceptron is an algorithm for supervised  learning of  binary classifiers,  which  was  invented in  1957  at the  Cornell Aeronautical Laboratory [Al_Janabi, Al_Shourbaji and Salman (2017)]. Initially seemed promising, the proception algorithm could not be trained to recognize many patterns for a dataset can be classified correctly if and only if the dataset itself is linear separable. 
      Due to the complex environment and large area of the forest, the stability and endurance of  the  WSN  are  required  generally.  Lora  (Long  Range)  technology  is  a  low-power wireless LPWAN protocol with long transmission distance and high security, which is one of  the most significant  factors that determines the efficiency of the entire system. Equipped  with  temperature,  humidity  and  wind  sensors,  Lora  terminals  deployed  in forests  collect weather  data and  transmit the  them  to server  through Lora  web gate. Weather parameters are obtained every 15 minutes, and if there is potential of forest fire, these parameters will be measured every 2 minutes, whose purpose is to reduce the usage of  battery  power. These  sensors  are powered  by  solar  panels  based  on  rechargeable technologies which converts solar into electricity to make sure our sensors could work regularly. 

![image](https://github.com/user-attachments/assets/6f2973a1-57ed-4f94-b617-d162d81cada6)

NONLINEAR PERCEPTRON (NP) 
We select the NP as a nonlinear benchmark model. The architecture of the NP model is comprised of a single neuron with a nonlinear activation function, whose input is {x i k }i∈{1,..., I} and output is y k. In addition, we set the activation function of NP to sigmoid function. 
MULTI-LAYER PERCEPTRON (MLP) 
The recent works have shown that MLP is a highly competitive model for multi-sensor fire detection. Thus, we select MLP as one of the neural network models against which the rTPNN will be compared. We use an MLP model which is comprised of H hidden layers and an output layer with a single neuron (namely, output neuron), where we set H = 3. We set the number of neurons nh = I. (H −h) at each hidden layer h ∈ {1, . . . , H − 1}, and nh = dI/2e at h = H. In addition, we set the activation function of each neuron at each layer to sigmoid function.

2.4	Proposed System
       The rTPNN model significantly differs from the existing methods due to recurrent sensor data processing employed in its architecture. rTPNN performs trend prediction and level prediction for the time series of each sensor reading and captures trends on multivariate time series data produced by multi-sensor detectors. We compare the performance of the rTPNN model with that of each of the Linear Regression (LR), Nonlinear Perceptron (NP), Multi-Layer Perceptron (MLP), Kendall- τ combined with MLP, Probabilistic Bayesian Neural Network (PBNN), Long-Short Term Memory (LSTM), and Support Vector Machine (SVM) on a publicly available fire data set. 
       Our results show that rTPNN model significantly outperforms all of the other models (with 96% accuracy) while it is the only model that achieves high True Positive and True Negative rates (both above 92%) at the same time. rTPNN also triggers an alarm in only 11 s from the start of the fire, where this duration is 22 s for the second-best model. Moreover, we present that the execution time of rTPNN is acceptable for real-time applications.
       The inputs of the rTPNN model are the reading of the sensors at discrete time k, { x i k}i∈{1,...,I} and that at k−1, {x i k−1 }i∈{1,...,I} , where I is the total number of sensors in the hardware implementation of fire detector. The output of rTPNN, y k is the state of the fire. At each discrete time k, y k takes value in the range [0, 1], where y k = 0 advocates that there is no fire and y k = 1 advocates otherwise. In the practical usage of rTPNN, if the value of y k is greater than a threshold γ , we state that there is fire.2 Otherwise, we state that there is no fire. In the architecture of rTPNN in Fig. 1, there is one Sensor Data Processing (SDP) module for each sensor i of fire detector, which is denoted by SDPi. At each discrete time k, for sensor i, the SDPi calculates the predicted trend t k i and the predicted level l k i of sensor i. The outputs of the SDPs are connected to the Fire Predictor (FP) module which predicts y k and is designed as fully connected layers.


2.5	Architecture

 ![image](https://github.com/user-attachments/assets/9bf2b3cb-0fcf-4525-82fd-819d2043671e)
Fig. 2.5.1 The architecture of the Recurrent Trend Predictive Neural Network (rTPNN).

![image](https://github.com/user-attachments/assets/53370c8f-5d5e-487d-9592-ce3c8c73b98b)
Fig. 2.5.2 The inner architecture of the Trend Predictor unit for sensor i. &  Fig. 2.5.3 The inner architecture of the Level Predictor unit for sensor i

3	DESIGN

3.1	Introduction
	       In this project, we evaluate the performance of the multi-sensor fire detector based on rTPNN for the 9 different real-life fire experiments from the available data set. For these experiments, for the multi-sensor detector, we compare the performance of our rTPNN model with that of each of the Linear Regression (LR), Non linear Perceptron (NP), Multi-Layer Perceptron (MLP), Kendall-τ combined with MLP (Kendall-MLP), Probabilistic Bayesian Neural Network (PBNN), Long-Short Term Memory (LSTM), and Support Vector Machine (SVM). We also compare the performance of rTPNN with those of the single-sensor detectors. While the used part of the publicly available data set contains 25796 samples, it provides meaningful performance evaluation results as well as a fair comparison of the models. Our results show that the rTPNN model outperforms all of the machine learning models while it achieves 
	1) prediction performance with high generalization ability
	2) low percentages for both FNR and FPR at the same time.
	3) early detection of fire
 
3.2	UML Diagram
![image](https://github.com/user-attachments/assets/48244978-3966-490c-a477-a38fbff06c89)
Fig.3.2.1

3.3	Data Set Descriptions
we evaluated the performance of the rTPNN model on the open-access data set available in the internet.
      This data set is comprised of 27 experiments each of which consists of the relative time to ignition in seconds as well as the sensor readings for an experiment. During these experiments, there are separately located sensors that measure the Temperature, Smoke Obscuration, and the concentrations of Carbon Monoxide, Carbon Dioxide, and Oxygen. however, the sensors that measure all of these metrics are only available in the (or close to the) bedroom in the testbed of data collection. Thus, we use only the experiments that are executed in the bedroom. 
       Moreover, since in this data set, three of the experiments (Experiment (Exp)-3, Exp-30, and Exp-32) are aborted due to the failure of ignition, we do not include those in our results. Thus, we use remaining 9 experiments.3 Accordingly, we use 10-fold cross-validation (CV) over the rest of the experiments for the training and test of the fire prediction methods. 
       Furthermore, in order to simplify the input set of each of the fire detection methods, we normalize each sensory data x k i as xik ← xik / maxk xik. 
       Note that while the original value of sensory data is greater than zero, the normalized data is in the range [0, 1]. In addition, for each experiment, we assume that the fire starts after 10 seconds (which is only two samples in the data set) from the ignition time. This assumption shows the expected time for the first triggered alarm but does not prevent fire detectors to detect fire earlier.
3.4	Methods and Algorithm
Linear regression
              You can use a fully connected neural network for regression, just don't use any activation unit in the end (i.e. take out the RELU, sigmoid) and just let the input parameter flow-out (y=x). Consider that a NN with one neuron without activation unit is basically a simple linear regression.
More generally this is not what you want, you will be using a very complicated structure of chained linear regressions that can tend to overfitting. Ridge regressions, Lasso, SVR and Bayesian predictors would do a better job at it and you would have better control about what you want to do.
Multilayer Perceptron (MLP)
              A multilayer perceptron (MLP) is a feedforward artificial neural network that generates a set of outputs from a set of inputs. An MLP is characterized by several layers of input nodes connected as a directed graph between the input and output layers. MLP uses backpropogation for training the network. MLP is a deep learning method.
              A multilayer perceptron is a neural network connecting multiple layers in a directed graph, which means that the signal path through the nodes only goes one way. Each node, apart from the input nodes, has a nonlinear activation function. An MLP uses backpropagation as a supervised learning technique. Since there are multiple layers of neurons, MLP is a deep learning technique.
       MLP is widely used for solving problems that require supervised learning as well as research into computational neuroscience and parallel distributed processing. Applications include speech recognition, image recognition and machine translation.
Long Short Term Memory (LSTM)
               Long Short Term Memory is a kind of recurrent neural network. In RNN output from the last step is fed as input in the current step. LSTM was designed by Hochreiter & Schmidhuber. It tackled the problem of long-term dependencies of RNN in which the RNN cannot predict the word stored in the long-term memory but can give more accurate predictions from the recent information. As the gap length increases RNN does not give an efficient performance. LSTM can by default retain the information for a long period of time. It is used for processing, predicting, and classifying on the basis of time-series data.
       Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN) that is specifically designed to handle sequential data, such as time series, speech, and text. LSTM networks are capable of learning long-term dependencies in sequential data, which makes them well suited for tasks such as language translation, speech recognition, and time series forecasting.
KENDALL-τ COMBINED WITH MLP (KENDALL-MLP) 
       The Kendall-MLP model is a fire detection method that uses the trend of the sensor reading. This method computes the trend via Kendall τ and predicts the fire via MLP based on the sensor readings and trend of those. Kendall-MLP achieves better accuracy than MLP and Radial Basis Function (RBF) neural network. In this project, for the MLP block in the Kendall-MLP model, we use the architecture which is described above.
PROBABILISTIC BAYESIAN NEURAL NETWORKS 
        For multi-sensor fire detection most researchers  have used the probabilistic neural networks which predict a distribution of fire probability at each time. In this project, we compare the performance of our rTPNN model against the Probabilistic Bayesian Neural Network (PBNN). The architecture of PBNN is comprised of a batch normalization layer, H = 3 hidden layers, a dense layer with two neurons and an output layer which returns a distribution. At each time, the fire probability is predicted as the mean of the distribution at the output of PBNN. We set the number of neurons nh = I.(H − h) at each hidden layer h ∈ {1, . . . , H − 1}, and nh = dI/2e at h = H. We also set the activation function of each neuron at each hidden layer to sigmoid. In addition, during the training of PBNN, the negative log likelihood is considered as the cost function.
