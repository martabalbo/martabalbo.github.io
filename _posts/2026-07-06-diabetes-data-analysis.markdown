---
layout: post
title:  "Diabetes Data Analysis"
date:   2026-07-06 11.54 +0100
categories: data analysis
---

# Warning: Website under construction

This is an analysis of a diabetes dataset, which I did after completing the courses on [Kaggle][kaggle]. Afterwards, I chose a dataset from [Kaggle datasets][kaggle-datasets], and performed a basic analysis using pandas to ... and scikit-learn to ...

This project was developed using Jupyter Notebooks. In this blog post I'll try to explain my thought process through every step of the project.
[Here][diabetes-data-analysis-github] you can find the original file.

# Project Overview

The dataset contains data collected by the Iraqi society, acquired from the laboratory of Medical City Hospital and the Specialized Center for Endocrinology and Diabetes-Al-Kindy Teaching Hospital. As said before, this dataset was sourced from [Kaggle][kaggle].

The columns in the dataset are:

* **ID**. Unique patient ID.
* **Gender**. Biological sex of the patient. Usually encoded as: 0 = Female, 1 = Male.
* **AGE**. Age of patient in years.
* **Urea**. Measures the level of urea in the blood (mg/dL). High levels may indicate kidney issues, common complications of diabetics. Normal range: 7-20 mg/dL.
* **Cr** (Creatinine). Measures the level of creatine in the blood (mg/dL). High levels may indicate kidney issues, common complications of diabetics. Normal range: 0.6-1.3 mg/dL.
* **HbA1c** (Glycated Hemoglobin). Indicator of average blood glucose levels over the past 2-3 months. Normal: 5.7%. Prediabetic: 5.7-6.4%. Diabetic: 6.5%.
* **Chol** (Cholesterol). Measures the level of cholesterol in the blood (mg/dL). High levels are a risk factor for cardiovascular disease, common complications of diabetics. Normal range: 200 mg/dL.
* **TG** (Triglycerides). Measures the level of fat in the blood (mg/dL). High levels is associated with insulin resistance and metabolic syndrome. Normal range: 150 mg/dL.
* **HDL** (High-Density Lipoprotein). Measures the level of "good" cholesterol (mg/dL). Helps remove excess cholesterol from the bloodstream. Ideal: 40 mg/dL (men), 50 mg/dL (women).
* **LDL** (Low-Density Lipoprotein). Measures the level of "bad" cholesterol (mg/dL). Contributes to plaque buildup in the arteries. Ideal: 100 mg/dL.
* **VLDL** (Very Low-Density Lipoprotein). Measures the level of another "bad" cholesterol (mg/dL), carries triglycerides. Often estimated from TG/5. High levels are associated with increased diabetes risk. Normal: 2-30 mg/dL.
* **BMI** (Body Mass Index). Measure of body fat based on height and weight (kg/m 
). Obesity is a risk factor for Type 2 diabetes. Underweight: 18.5, Normal: 18.5-24.9, Overweight: 25-29.9, Obese: 30.
* **CLASS**. Indicates whether the patient is Non-Diabetic ('N'), is Diabetic ('Y') or is Predict-Diabetic ('P').

**Source:** Rashid, Ahlam (2020), “Diabetes Dataset”, Mendeley Data, V1, doi: 10.17632/wj9rwkp9c2.1

I started by importing the necessary libraries.

{% highlight python %}

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

{% endhighlight %}

After that, I loaded the dataset using pandas.

{% highlight python %}

def load_data(path='Multiclass Diabetes Dataset - Original.csv'):
    df = pd.read_csv(path)

    df.drop(columns=['No_Pation'], inplace=True)
    
    return df

{% endhighlight %}

{% highlight python %}

df = load_data()

print(df)
print(df.columns)

{% endhighlight %}

This is the resulting dataset:

{% highlight python %}

            ID Gender  AGE  Urea  Cr  HbA1c  Chol   TG  HDL  LDL  VLDL   BMI CLASS
      0    502      F   50   4.7  46    4.9   4.2  0.9  2.4  1.4   0.5  24.0     N
      1    735      M   26   4.5  62    4.9   3.7  1.4  1.1  2.1   0.6  23.0     N
      2    420      F   50   4.7  46    4.9   4.2  0.9  2.4  1.4   0.5  24.0     N
      3    680      F   50   4.7  46    4.9   4.2  0.9  2.4  1.4   0.5  24.0     N
      4    504      M   33   7.1  46    4.9   4.9  1.0  0.8  2.0   0.4  21.0     N
      ..   ...    ...  ...   ...  ..    ...   ...  ...  ...  ...   ...   ...   ...
      995  200      M   71  11.0  97    7.0   7.5  1.7  1.2  1.8   0.6  30.0     Y
      996  671      M   31   3.0  60   12.3   4.1  2.2  0.7  2.4  15.4  37.2     Y
      997  669      M   30   7.1  81    6.7   4.1  1.1  1.2  2.4   8.1  27.4     Y
      998   99      M   38   5.8  59    6.7   5.3  2.0  1.6  2.9  14.0  40.5     Y
      999  248      M   54   5.0  67    6.9   3.8  1.7  1.1  3.0   0.7  33.0     Y

      [1000 rows x 13 columns]
      Index(['ID', 'Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL',
            'LDL', 'VLDL', 'BMI', 'CLASS'],
            dtype='object')

{% endhighlight %}

# Project Questions and Objectives

I choose two objectives, ones that allowed me to work with both data analysis and machine learning.

The first part was about data analysis. Considering age, gender, BMI, blood sugar level and cholesterol, which of these factors are risk factors for Type 2 diabetes?

The second used machine learning: develop a machine learning model that can predict diabetes status (Diabetic, Non-Diabetic or Predict-Diabetic).

# Data Cleaning and Pre-Processing

The first step was making the data easier to work with.

I did it by removing duplicates and null values. There were no null values, as you can see with this function.

{% highlight python %}

def clean_data(df):
    print(df.isna().any())
    df.dropna(inplace=True)

    df.drop_duplicates(['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'CLASS'], inplace=True)

    df.reset_index(drop=True, inplace=True)

clean_data(df)

{% endhighlight %}

{% highlight python %}

ID        False
Gender    False
AGE       False
Urea      False
Cr        False
HbA1c     False
Chol      False
TG        False
HDL       False
LDL       False
VLDL      False
BMI       False
CLASS     False
dtype: bool

{% endhighlight %}

Furthermore, to render the data easier to analyze, I decide to encode the 'Gender' and 'CLASS' after checking how many values appear in them.

{% highlight python %}

def data_exploration(df):
    # find problems in categorical rows
    print(df['Gender'].unique()) 
    print(df['CLASS'].unique())
    
    # find how big are problems in categorical rows
    print(df['Gender'].value_counts())
    print(df['CLASS'].value_counts())

data_exploration(df)

{% endhighlight %}

{% highlight python %}

['F' 'M' 'f']
['N' 'N ' 'P' 'Y' 'Y ']
Gender
M    467
F    363
f      1
Name: count, dtype: int64
CLASS
Y     691
N      95
P      40
Y       4
N       1
Name: count, dtype: int64

{% endhighlight %}

Since the unexpected values are relatively few, I simply dropped those from the database.

I decided to encode the columns with F = 0 and M = 1 for 'Gender', and with N = 0, P = 1, Y = 2 for 'CLASS'.


{% highlight python %}

def pre_process_data(df):
    # drop 'N ' and 'Y ' values from 'CLASS', and 'f' values from 'Gender' (they are a small enough number)
    df.drop(df[(df['CLASS']=='Y ') | (df['CLASS']=='N ') | (df['Gender']=='f')].index, inplace=True)
    
    # encode 'Gender' and 'CLASS' columns: [F=0, M=1] and ['N'=0, 'P'=1, 'Y'=2]
    enc = OrdinalEncoder()
    df[['Gender', 'CLASS']] = enc.fit_transform(df[['Gender', 'CLASS']])
    
    return df

pre_process_data(df)

{% endhighlight %}

{% highlight python %}

	ID	Gender	AGE	Urea	Cr	HbA1c	Chol	TG	HDL	LDL	VLDL	BMI	CLASS
0	502	0.0	50	4.7	46	4.9	4.2	0.9	2.4	1.4	0.5	24.0	0.0
1	735	1.0	26	4.5	62	4.9	3.7	1.4	1.1	2.1	0.6	23.0	0.0
2	504	1.0	33	7.1	46	4.9	4.9	1.0	0.8	2.0	0.4	21.0	0.0
3	634	0.0	45	2.3	24	4.0	2.9	1.0	1.0	1.5	0.4	21.0	0.0
4	721	0.0	50	2.0	50	4.0	3.6	1.3	0.9	2.1	0.6	24.0	0.0
...	...	...	...	...	...	...	...	...	...	...	...	...	...
821	194	0.0	57	4.1	70	9.3	5.3	3.3	1.0	1.4	1.3	29.0	2.0
823	196	1.0	55	3.1	39	8.5	5.0	2.5	1.9	2.9	0.7	27.0	2.0
824	198	1.0	28	3.5	61	8.5	4.5	1.9	1.1	2.6	0.8	37.0	2.0
825	199	1.0	69	10.3	185	7.7	4.9	1.9	1.2	3.0	0.7	37.0	2.0
826	200	1.0	71	11.0	97	7.0	7.5	1.7	1.2	1.8	0.6	30.0	2.0
825 rows x 13 columns

{% endhighlight %}

# Exploratory Data Analysis (EDA)

I wrote a section to visualize the distribution of the factors I was considering (age, gender, BMI, blood sugar level and cholesterol), both on the whole population and by class. I chose histograms to visualize the distribution, and boxplots to see whether the distribution is simmetrical or skewed.

## Gender distribution

{% highlight python %}

def plot_gender(df):
    # visualize the correlation between gender and class
    df_reduced = df[['Gender', 'CLASS']]

    # plot the gender distribution
    
    df1 = df_reduced.groupby(['Gender']).size().reset_index(name='number_patients')

    plt.figure(figsize=(5, 5))
    ax = sns.barplot(x='Gender', y='number_patients', hue='Gender', data=df1)
    plt.ylabel("Number of Patients")

    legend = ax.get_legend()
    handles = legend.legend_handles
    ax.legend(handles, ["Female","Male"], title='Gender')

    ax.set_xticks(range(2))
    ax.set_xticklabels(['Female', 'Male'])
    
    # plot the gender distribution by class

    df2 = df_reduced.groupby(['Gender', 'CLASS']).size().reset_index(name='number_patients')

    plt.figure(figsize=(5, 5))
    ax = sns.barplot(x='CLASS', y='number_patients', hue='Gender', data=df2)
    plt.ylabel("Number of Patients")
    
    legend = ax.get_legend()
    handles = legend.legend_handles
    ax.legend(handles, ["Female","Male"], title='Gender')

    ax.set_xticks(range(3))
    ax.set_xticklabels(['non-diabetic', 'predict-diabetic', 'diabetic'])

plot_gender(df)

{% endhighlight %}

{% highlight python %}

      "image/png": "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",


{% endhighlight %}

{% highlight python %}

{% endhighlight %}

{% highlight python %}

{% endhighlight %}

{% highlight python %}

{% endhighlight %}



{% highlight python %}

{% endhighlight %}

[kaggle]: https://www.kaggle.com/
[kaggle-datasets]: https://www.kaggle.com/datasets
[diabetes-data-analysis-github]: https://github.com/martabalbo/diabetes-data-analysis
