# Project 4: SafeComm Digital Security Solutions 
Federico de Nuñez, Valentina Pancaldi, Arthur Noel Birnstiel




## Dataset features
• Fraudulent: Binary indicator if the SMS is fraudulent (1 for Yes, 0 for No)

• SMS Text: The content of the SMS

• ID: A unique identifier for each SMS

• Date and Time: Timestamp indicating when the SMS was sent

## Assignment
• Perform an Explanatory data analysis (EDA) with visualization using the entire dataset..

• Preprocess the dataset (impute missing values, encode categorical features with one-hot
encoding). Your goal is to estimate whether an SMS is fraudulent

• Define whether this is a regression, classification or clustering problem, explain why and
choose your model design accordingly. Test at least 3 different models. First, create a
validation set from the training set to analyze the behaviour with the default
hyperparameters. Then use cross-validation to find the best set of hyperparameters. You
must describe every hyperparameter tuned (the more, the better)

• Select the best architecture using the right metric

• Compute the performances of the test set
s
• Explain your results



## Introduction – Briefly describe your project
Welcome to SafeComm Digital Security Solutions! In the modern digital age, people across the globe
communicate largely through text messages. SMSs have become an integral part of our daily lives.
However, with this ease of communication, there comes a dark side: SMS-based fraud. Unsuspecting
individuals often receive malicious or scam texts intending to deceive or cause harm.
SafeComm has recently partnered with a major telecom provider that has shared anonymized SMS
data. This dataset comprises a mix of regular day-to-day messages and some potentially fraudulent
ones. The objective is to design a mechanism that identifies and flags these fraudulent messages
automatically. This way, we can warn users or even prevent these messages from being delivered
altogether.

## Methods – Describe your proposed ideas (e.g., features, algorithm(s),
training overview, design choices, etc.) and your environment so that:
• A reader can understand why you made your design decisions and the
reasons behind any other choice related to the project
• A reader should be able to recreate your environment (e.g., conda list,
conda envexport, etc.)
• It may help to include a figure illustrating your ideas, e.g., a flowchart
illustrating the steps in your machine learning system(s)

## Experimental Design – Describe any experiments you conducted to
demonstrate/validate the target contribution(s) of your project; indicate the
following for each experiment:
• The main purpose: 1-2 sentence high-level explanation
• Baseline(s): describe the method(s) that you used to compare your work
to
• Evaluation Metrics(s): which ones did you use and why?

## Results – Describe the following:
• Main finding(s): report your final results and what you might conclude
from your work
• Include at least one placeholder figure and/or table for communicating
your findings
• All the figures containing results should be generated from the code.

## Conclusions – List some concluding remarks. In particular:
• Summarize in one paragraph the take-away point from your work.
• Include one paragraph to explain what questions may not be fully
answered by your work as well as natural next steps for this direction of
future work
