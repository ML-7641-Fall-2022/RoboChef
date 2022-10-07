# RoboChef
This repository contains source code for ML-7641 Fall-22 Project


## Background
Machine Learning systems today aid decision making across the entrire spectrum of life. One of such relevant decision, that each human has to make daily is his/her Meal Choice. The said decision depends on a multitude of factors like nutritional requirements, eating prefrences and mood etc. Thus Meal Choice presents lot of variability not only in between two different indviduals but also for the same indvidual.

In this project we thus aim to build a ML system, that aids multiple aspects of Meal Choice decision.

## Problem Definition
We want to tackle two main problems here:

1. **Given an image of a food item, what are the key ingredients in the food?**

2. **Given my previous interaction with food items, can you recommend me similar foods?**
    * Can the reccomendation also account for additional constraints from user like nutritional requirements etc?


### Usecases
The modular nature of above problems lends itself to multiple usecases:
* When encountered with an unknown food item, the user might want to know its ingredients, such that he can figure out:
    - Is the dish healthy?
    - If the dish contains allergens?

* As a next step, user might want reccomendations for similar dishes that he took image of. Additionally, there can be numerous additional constraints placed on the reccomendations recieved like:
    - Restrict reccomendations to ingredients he already has
    - Constrain by the nutritional requirements of the dish, say the user wants only "low calorie" dishes.


## Methodology
We divide our project pipeline into three main stages:

1. **Classfication System**: This stage would take as input food images and passes them through a CNN to output food labels. The output of this stage is (**Output 1**), and flows to next stage, but can also be used independently.

2. **Food to Ingredient Mapping**: Output 1 from CNN is then querried through a database housing the mapping between food items and their respective ingredients, to yeild **Output 2**. This can again either be used independtly or overlaid over reccomendation system as a filter.

3. **Reccomendation System**: In this stage we reccommend the user additional food items: **Output 3** basis his (&other individuals) interactions. We also allow for the user to place additional constraints over the reccomendations.

The schematic of these stages is given below:
![Getting Started](./images/ml_project_pipeline.jpg)

### CNN Module
We will be training the Food-101 dataset using pre-trained CNN architectures like DenseNets and ResNets to classify the image into one of the 101 categories. To evaluate the model performance, we will be using top-1 and top-5 classification accuracy on the predictions.

### Reccomendation System Module

## Data Collection
### Data for Recommendation system
[Dataset Source](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv)

In this part, we’re using data from the Food.com Recipes and Interactions, which contains 2 datasets: user interactions data and recipe dataset 
   - user interactions dataset: This dataset contains the user Id and users’ interaction with recipes, such as rating and review. 
   - recipe dataset: This dataset contains the recipe information recipe ID, nutrition information, steps to cook, time to cook, etc.

We will join user interactions dataset and recipe dataset based on recipe_id. With the joined data, we’ll use our recommendation system to study user preferences and recommend recipes to users based on their previous behaviors, and optionally input from the ingredient we get from the classification task.

## Results & Discussion

## Conclusion

## Refrences

<a id="Trang">[1]</a> 
Trang Tran, T.N., Atas, M., Felfernig, A. et al. An overview of recommender systems in the healthy food domain. J Intell Inf Syst 50, 501–526 (2018). https://doi.org/10.1007/s10844-017-0469-0

<a id="2">[2]</a>
Kaggle Data set for User food Interactions
https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv

<a id="3">[2]</a> 

## Project Logistics
### Contribution table
### GANTT Chart
