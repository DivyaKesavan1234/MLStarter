# PROJECT TITLE 


## NON-TECHNICAL EXPLANATION OF YOUR PROJECT
Imagine we have a big bag of information about houses in Boston, like how many rooms they have, how far they are from work, and other details.
And you also know how much each house is worth.Now, you want to use this information to create a magical machine that can predict the price of any house in Boston, 
even if you've never seen it before. This way, if you're interested in buying or selling a house, you can get a pretty good idea of how much it might cost.
First, we will teach the machine to understand the patterns in the information you have about the houses and how they are priced. 
We'll show it many examples of houses along with their prices, so it learns these patterns really well.
Once the machine learns these patterns, we'll give it a new set of information about a house that you want to predict the price for. 
The model will use what it learned to make a smart guess about how much that house might be worth.
But the machine isn't perfect. Sometimes, it might guess too high or too low. So, we'll use some special methods to check how accurate its
guesses are. If the model making a lot of mistakes, we'll go back and teach it better, helping it understand the patterns even more. As 
we keep fine-tuning the machine and making it smarter, its guesses about house prices will get better and better. And that's the goal – 
to create a machine learning model that can help people figure out the prices of houses in Boston just by looking at some information!

## DATA
The dataset we're using is called the "Boston Housing Dataset." It's a collection of information about houses in the city of Boston. 
This dataset includes various details about each house, like the average number of rooms, crime rates in the area, distance to employment centers, and more. 
Each house's price is also included in the dataset.

## MODEL 
In this project where we're predicting Boston house prices, we're using a machine learning model called a "Gradient Boosting Regressor."

Here's a summary of the model and why we chose it:

Model: Gradient Boosting Regressor

A Gradient Boosting Regressor is a type of machine learning algorithm that's really good at predicting numerical values, like the prices of houses.
It's a combination of many simpler models called "decision trees." These trees are like a series of questions that help the model make predictions.
The model learns by looking at the errors (the differences between its predictions and the actual prices) and trying to correct them step by step. 
It keeps getting better with each correction.
Why We Chose It:

Gradient Boosting is known for its strong predictive performance. It can capture complex relationships in the data.
It handles feature interactions well, meaning it can understand how different features work together to affect house prices.
It's less prone to overfitting (making predictions that work well on training data but not on new data), which is important to make sure our 
predictions are accurate for new houses.
The Boston Housing Dataset isn't too large, so the extra computational cost of this algorithm isn't a big concern.
In short, we chose the Gradient Boosting Regressor because it's powerful, accurate, and can handle the kind of data we have about Boston houses, which has various features influencing the prices.
 

## HYPERPARAMETER OPTIMSATION

**Number of Estimators (Trees):**
This hyperparameter controls the number of decision trees in the boosting process.
Higher values can lead to a more complex model but may risk overfitting.
Optimization: You might use techniques like grid search or random search to try different values and see which gives the best performance on a validation set.

**Learning Rate:**
The learning rate controls how much each new tree tries to correct the mistakes of the previous trees.
Lower values make the learning process slower but might result in better convergence.
Optimization: You can experiment with different learning rates to find a balance between training speed and model performance.

**Maximum Depth of Trees:**
This sets the maximum number of questions (splits) a decision tree can have.
Deeper trees can capture more complex patterns but may lead to overfitting.
Optimization: You might search over a range of depths to find the optimal trade-off between model complexity and performance.
**Minimum Samples per Leaf or Split:**
These parameters control the minimum number of samples required to create a new split or form a leaf node.
Larger values prevent the model from creating overly specific rules based on small amounts of data.
Optimization: You could try different values to ensure the model doesn't overfit by creating too specialized rules.

**Subsample Ratio:**

This hyperparameter determines the fraction of samples used for training each tree. It's used to introduce randomness and reduce overfitting.
Lower values can help prevent overfitting but might lead to a decrease in model performance.
Optimization: Experiment with different subsample ratios to find the balance between randomness and predictive accuracy.
**Loss Function:**

The loss function measures the difference between the model's predictions and the actual target values.
The choice of loss function depends on the problem at hand; for regression, Mean Squared Error (MSE) is commonly used.
Optimization: The loss function is usually fixed based on the problem type.


## RESULTS
	Model	R-squared Score
1	Random Forest	86.406033
2	XGBoost	84.948947
0	Linear Regression	71.218184
3	Support Vector Machines	59.001585

What we Can Learn from the Model:

Performance Assessment:

The MSE values on the training, validation, and test data provide insight into how well your model is performing.
The lower the MSE, the better your model is at making accurate predictions. An MSE of 11,500 on the test data suggests that the model is performing reasonably well.
Overfitting Evaluation:

Comparing the MSE on the training data to the validation and test data can help you understand if your model is overfitting.
If the training MSE is significantly lower than the validation and test MSE, it indicates that the model might be overfitting to the training data.
Generalization Capability:

The similar MSE values on the validation and test data suggest that your model is generalizing well to unseen data.
This is a positive sign, indicating that your model has learned the underlying patterns in the data and is making consistent predictions on new examples.
Feature Importance:

The model might provide information about the importance of different features in predicting house prices.
You can learn which features have the most influence on the predictions. For instance, features like average number of rooms and crime rate might have high importance.
Areas for Improvement:

Analyzing the errors the model makes can provide insights into the types of houses it struggles to predict accurately.
You can use this information to refine your feature engineering, try different algorithms, or experiment with additional data preprocessing steps.
Model Complexity:

The model's performance on the validation and test data helps you gauge the right balance between model complexity and generalization.
If the validation and test errors are close, it suggests that the model isn't underfitting (too simple) or overfitting (too complex).
Validation Strategy Effectiveness:

If the validation and test errors are significantly different, it might indicate that your validation strategy needs adjustments.
Revisiting your validation technique could help ensure a more accurate estimate of your model's performance.

You can include images of plots using the code below:
![Screenshot](image.png)[image](https://github.com/DivyaKesavan1234/MLStarter/assets/142207537/7122b004-dd87-439d-be74-0abb360ae511)



