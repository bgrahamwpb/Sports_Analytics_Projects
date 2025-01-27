**Baseball Swing Probability Analysis**


**Overview**

This repository contains a data science project focused on predicting swing probabilities for pitches in baseball games. The dataset comprises information from approximately 2,000,000 pitches over three seasons. The primary goal is to predict swing probabilities for pitches in the third season of the dataset, which lacks the description column indicating swing events.



**Repository Contents**

-year1.csv: Dataset for the first season.

-year2.csv: Dataset for the second season.

-year3.csv: Original dataset for the third season without swing probabilities.

-validation.csv: Dataset with appended swing probabilities for the third season.

-swing_probability_modeling_final.py: Script used for data preprocessing, model training,   and predictions.

-swing_probability_modeling_final.ipynb: Jupyter Notebook containing the analysis,  visualizations, and model evaluation.

-writeup.pdf: Detailed methodology and assumptions.

-swing_probabilities_plots.pdf: Visualizations of swing probabilities.

-documentation.csv: Definitions of the columns in the dataset.



**Prerequisites**

Python 3.x, Jupyter Notebook, pandas, numpy, matplotlib, seaborn, scikit-learn



**Usage**

1. Clone the Repository:

         git clone https://github.com/bgrahamwpb/Sports_Analytics_Projects.git

         cd Sports_Analytics_Projects


2. Install Dependencies

         pip install pandas, numpy, matplotlib, seaborn, scikit-learn

3. Run the Script

         python swing_probability_modeling_final.py

4. Explore the Jupyter Notebook

         Open swing_probability_modeling_final.ipynb in Jupyter Notebook




**Methodology**

Detailed methodology, feature engineering, model training, and evaluation processes are described in writeup.pdf.



**Visualizations**

Swing probability visualizations can be found in swing_probabilities_plots.pdf.



