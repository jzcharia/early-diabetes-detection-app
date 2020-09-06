# Early Diabetes Detection Web Application - Project Overview
  * Built a random forest model to help detect diabetes early based on 16 unique features (R^2 = 96%)
  * Created a web application to interact with model and provide live predictions using StreamLit
  * Pushed the web application to the cloud (hosted by Salesforce's Heroku)

## Resources

**Major packages used:** Pandas, Numpy, Scikit-Learn, Streamlit, Matplotlib, Seaborn, Pickle

**Source of Data:** Islam, MM Faniqul, et al. 'Likelihood prediction of diabetes at early stage using data mining techniques.' Computer Vision and Machine Intelligence in Medical Image Analysis. Springer, Singapore, 2020. 113-125. (https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.)

**StreamLit Production:** https://github.com/dataprofessor/penguins-heroku/blob/master/penguins-app.py

**Tools:** Jupyter Notebook, Anaconda Prompt, Heroku

# Future Improvements
 * **Better Dataset -** the sample was not great. The data was sourced from Bangladesh which has a population that does not represent the USA. Specifically, we can see that in the dataset when looking at obesity and genital thrush rates. 
 * **More Interactive Features -** the web application is pretty plain. It allows you to answer some questions and it returns a prediction. Additional features to be added:
    1. A window to show actual probablity of event.
    2. A submit button -- currently it updates every time a change is made
    3. Resources for diabetes help to come up when the prediction is yes
 * **Not a binary target-** options for at high risk or low risk would be a great way to show more subtly to the app. This can be based on probability or a new dataset to train for that. 
