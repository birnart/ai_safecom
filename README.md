

# Project 4: SafeComm Digital Security Solutions 

 Team members: Arthur Birnstiel, Valentina Pancalid and Federico de Nuñez

## Section 1

In the modern digital age, people across the globe
communicate largely through text messages. SMSs have become an integral part of our daily lives.
However, with this ease of communication, there comes a dark side: SMS-based fraud. Unsuspecting
individuals often receive malicious or scam texts intending to deceive or cause harm.
We are SafeComm Digital Security Solutions, a consulting firm asked to design a mechanism that identifies and flahss fraudulent SMS messages automatically to prevent users or prevent these messages from being delivered altogether. 


Dataset features: 

• Fraudulent: Binary indicator if the SMS is fraudulent (1 for Yes, 0 for No) (int)

• SMS Text: The content of the SMS (string)

• ID: A unique identifier for each SMS (int)

• Date and Time: Timestamp indicating when the SMS was sent (DateType)

## Section 2

We propose creating machine learning models and train them to analyse sms messages and decide if they are spam. In short, we propose to create a classifier. In order to create the best classifier, we will test different algorithms and decide on the best. We chose to work with: Logistic Regression, SVM and kernel SVM, Multinomial Naive Bayes and an Artificial Neural Network. 

We received a dataset with messages in form of text and had to convert it into numeric values for the algorithms to process. 
First we worked with our text, we imported typical stopwords and removed them, removed html strips, removed any characters that aren't lower-case letters and finally stemmed the text. The final result of this treatment is that we only keep the most important roots of the words.
Then we used two different text to values converter, Term Frequency - Inverse Document Frequency and Bag of Words (which we will call TF-IDF and BoW from this point onwards). 



## Section 3
Some of the algorithms listed require or greatly benefit with normalized or standarized data, the TF-IDF output is already normalized but the BoW is not, this is why we differenciate the normalized BoW data and the not normalized BoW natural output. 

For our experimentation process we will first test each model with default/non tuned hyperparameters to see their results. To run a model, we will train it with our training data, then predict using our X_test data and finally we will compare the predicted results with the real Y_test results to create a confusion matrix. 

We will then tune the hyperparameters using cross-validation to find the best ones. 

    Common hyperaparameter: 
        • max_iter: Some of the models we use have this hyperparameter, this parameter determines the maximun number of iterations the model will do before finishing the execution. 

    Logistic Regression:
        •  penalty: this hyperparameter specifies the norm of the penalty which is used to prevent overfitting by adding a penalty term for complex models.
        •  C: Inverse of regularization strenght, the smaller the float, the stronger the regularization. 

    Linear SVM with stochastic gradient descent training:
        • loss: the loss function to be used which quantifies the difference between the predicted values and the true labels, the goal of training is to minimize this loss
        • learning_rate: the learning rate to apply to the SGD which controls the size of the steps taken during each iteration of the optimization process. It determines how much the model's parameters are adjusted based on the gradient of the loss function with respect to those parameters.

    Kernel SVM:
        • kernel: this hyperparameter specifies the type of kernel function to be used in the algorithm. It transforms the input data into a higher-dimensional space and therefore it allows the algorithm to find nonlinear decision boundaries in the input space.
        • gamma: kernel coefficient for the method we decide on kernel. It influences the shape of the decision boundary and the influence of individual training samples.

    Multinomial Naive Bayes:
        • alpha: additive smoothing parameter, it is used to handle the issue of zero probabilities for certain features in the training data.

    Artificial Neural Network:
        • kernel initializer:specifies the method used to initialize the weights of the neural network layers.
        • activation functions: mathematical operations applied to the output of each neuron. They introduce non-linearity to the network, allowing it to learn complex relationships and patterns in the data
        • optimizer: defines the specific optimization algorithm used to update the weights during the training process
        • loss: the loss function to be used which quantifies the difference between the predicted values and the true labels, the goal of training is to minimize this loss
        • epochs: by the 5th epoch whe can see that the change is the loss is not large so it is sufficient and the accuracy is way higer so we could be doing overfitting
        



Both for evaluating our models and for deciding the best hyperparameters we use *f1* scoring because we have an unbalanced dataset as can be seen in the visualization and the accuracy may mislead us due to the dominance of one class (in our case the Non Fraudulent sms). 
Furthermore, we want to minimize False Positives because we don't want to accidentally flag a non fraudulent sms as spam. 

For testing, we will redo the whole process of training and running the models but with the best hyperaparameters found.


## Section 4:  

Finally, we conclude our experimentation after executing the models with these confusion matrixes. As we tuned our hyperparameters maximizing the f1 score, some results may seem worse but actually make the f1 grow in comparison to the default models. 

These are the confusion matrices of the default or non specific hyperparameters models:



These are the confusion matrices of the hyperparameter tuned models:





## Section 5: 

After considering multiple models, training them and tuning their hyperparameters, we conclude that the tuned Kernel SVM is the best architecture to implement for our problem. It has high recall and low false positives as can be seen on the second image on section 4 and more importantly, the highest F1 score out of all the models we tried during our experimentation. 

Even though we tuned with high details our models and compared the best performance of each, we still could have created an even more powerfull classifier by chaining them and creating an ensemble of classifiers that combine the best qualities and predictions of our models. The next step for this direction of future work will be to implement it and tune to compare if a composition of classifiers is better than single models. 

