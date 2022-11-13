# RoboChef

[comment]: <> (TODO)
<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project"> ➤ About The Project</a></li>
    <li><a href="#prerequisites"> ➤ Prerequisites</a></li>
    <li><a href="#folder-structure"> ➤ Project Organsisation</a></li>
    <li><a href="#dataset"> ➤ Dataset</a></li>
    <li><a href="#roadmap"> ➤ Roadmap</a></li>
    <li><a href="#contributors"> ➤ Contributors</a></li>
  </ol>
</details>

## Background
Machine Learning systems today aid decision making across the entire spectrum of life. One such decision, that each human has to make daily is his/her "meal choice" . The said decision depends on a multitude of factors like nutritional requirements, eating preferences and mood etc.

In this project, we thus aim to build an ML system that aids multiple aspects of Meal Choice decision using Image Recognition and Recommender Systems.

## Problem Definition
We want to tackle two main problems here:

1. **Given an image of a food item, what are the key ingredients that goes in preparing the food item ?**

2. **What food items should we recommend to a user based on his preferences and other similar users food choices?**
   - Can the recommendation also account for additional constraints from user like nutritional requirements, calories level, etc?


### Usecases
The modular nature of above problems lends itself to multiple usecases:
* When encountered with an unknown food item, the user might want to know its ingredients, such that he can figure out:
    - Is the dish healthy?
    - If the dish contains allergens?

* As a next step, user might want recommendations for similar dishes that he took image of. Additionally, there can be numerous additional constraints placed on the recommendations recieved like:
    - Restrict recommendations to ingredients he already has
    - Constrain by the nutritional requirements of the dish, say the user wants only "low calorie" dishes.


## Project Pipeline
We divide our project pipeline into three main stages:

1. **Classfication System**: This stage would take as input food images and passes them through a CNN to output food labels. The output of this stage is (**Output 1**), and flows to next stage, but can also be used independently.

2. **Food to Ingredient Mapping**: Output 1 from CNN is then queried through a database housing the mapping between food items and their respective ingredients, to yield **Output 2**. This can again either be used independently or overlaid over recommendation system as a filter.

3. **Recommendation System**: In this stage we reccommend the user additional food items: **Output 3** basis his (&other individuals) interactions. We also allow for the user to place additional constraints over the recommendations.

The schematic of these stages is given below:
![Getting Started](./images/ml_project_pipeline.jpg)


## Data Collection
### Data for CNN Module
[Dataset Source](https://www.kaggle.com/datasets/dansbecker/food-101)

We will be using Food-101 dataset for the CNN classification which consists of 101 food categories with a total of 101,000 images. 

### Data for Recommendation system
[Dataset Source](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv)

In this part, we’re using data from the Food.com Recipes and Interactions, which contains 2 datasets: user interactions data and recipe dataset 
   - user interactions dataset: This dataset contains the user Id and users’ interaction with recipes, such as rating and review. 
   - recipe dataset: This dataset contains the recipe information recipe ID, nutrition information, steps to cook, time to cook, etc.

We will join user interactions dataset and recipe dataset based on recipe_id. With the joined data, we’ll use our recommendation system to study user preferences and recommend recipes to users based on their previous behaviors, and optionally input from the ingredient we get from the classification task.


## Results & Discussion
As defined in the pipeline above, there are three main modules of our ML system, and this section details the exect semantics of the work on each of the items.

### CNN System Module
### Food Label to Ingredient Mapping
### Recommendation System Module
We will use the user-food interaction data which contains the temporal food-item ratings given by users to provide recommendations for similar food items leveraging the user-user collaborative filtering and matrix factorization techniques.

![Collabartive Filtering](./images/collaborative_filtering.png?raw=true)

#### Exploratory Data Analysis
##### User Recipe Interaction Data
The user interaction recipe data has 5 columns, with head of the table given below:

```bash
Index(['user_id', 'recipe_id', 'date', 'rating', 'review'], dtype='object')
```
| user_id | recipe_id |       date | rating |                                            review |
|--------:|----------:|-----------:|-------:|--------------------------------------------------:|
|   38094 |     40893 | 2003-02-17 |      4 | Great with a salad. Cooked on top of stove for... |
| 1293707 |     40893 | 2011-12-21 |      5 | So simple, so delicious! Great for chilly fall... |
|    8937 |     44394 | 2002-12-01 |      4 |  This worked very well and is EASY. I used not... |
|  126440 |     85009 | 2010-02-27 |      5 | I made the Mexican topping and took it to bunk... |
|   57222 |     85009 | 2011-10-01 |      5 | Made the cheddar bacon topping, adding a sprin... |

The number of unique users and unique recipes is given as:
```bash
user_id         226570
recipe_id       231637
user_recipe    1132367
```

<span style="color:red">As expected not every user rates every recipe, which is apparent from the counts above. An estimate of the sparsity of interaction matrix is:</span>
```python
sparsity = 1- (1132367/(N_users*N_Recipes))
print (f"Sparsity in data {sparsity:.9%}")
#Sparsity in data 99.997842371%
```

We have analysed the distribution of these interactions below:

1a. How many recipes do the users rate?
```python
user_grp[[("recipe_id","count")]].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
```
|      Percentile | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |    1.0 |
|----------------:|----:|----:|----:|----:|----:|----:|----:|----:|----:|-------:|
| recipe_id,count | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 2.0 | 5.0 | 7671.0 |

<span style="color:red">Thus almost 90% of the users rate <=5 recipes, to create a heavy left tail skew.</span>

1b. How many users rate the same recipes ?
The converse of the above distribution is the distribution of users rating the same recipe.
```python
recipe_grp[[("user_id","count")]].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
```
| Percentile    | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0    |
|---------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|--------|
| user_id,count | 1.0 | 1.0 | 1.0 | 2.0 | 2.0 | 3.0 | 3.0 | 5.0 | 9.0 | 1613.0 |

<span style="color:red">Similar to above we see a highly skewed distribution, with 80% of the recipes being rated by <=5 users</span>

2. Distribution of Ratings?
```python
raw_interactions["rating"].hist()
```
![Ratings Histogram](./images/rating_histogram.png?raw=true)

<span style="color:red">The ratings follow a heavy skew, with 4 and 5 being the predominant rating</span>


#### Modelling
We have tried two approaches for recommendation system:
##### 1. Collaborative Filtering
##### 2. Matrix Factorisation
The matrix factorization method will use the concept of Singular Value Decomposition to obtain highly predictive latent features using the sparse ratings matrix and provide a fair approximation of predictions of new items ratings.
###### Analysis of latent features


## Results & Discussion (old)

### CNN System Module

We will be training the Food-101 dataset using pre-trained CNN architectures like DenseNets and ResNets to classify the image into one of the 101 categories. 
- Evaluation metrics to be used: **top-1** and **top-5** classification accuracy on the predictions.

### Recommendation System Module

We will use the user-food interaction data which contains the temporal food-item ratings given by users to provide recommendations for similar food items leveraging the user-user collaborative filtering and matrix factorization techniques.

![Collabartive Filtering](./images/collaborative_filtering.png?raw=true)

The matrix factorization method will use the concept of Truncated Singular Value Decomposition to obtain highly predictive latent features using the sparse ratings matrix and provide a fair approximation of predictions of new items ratings.

In recommendation systems , we have to not only ensure greater accuracy on ratings prediction but also have the most relevant items at the top of the recommendation list i.e. ranking of the recommendations.
- Evalution metrics to be used : **MAP@k** (Mean Average Precision at K) and **NDCG** (Normalized Discounted Cummulative Gain)

### Points for further exploration

- Applications of autoencoders to learn underlying feature representation and provide a more personalized recommendation.
- Added functionality to recommend the food items that can be prepared using the ingredients image a user has uploaded.

## References

<a id="Trang">[1]</a> 
Trang Tran, T.N., Atas, M., Felfernig, A. et al. An overview of recommender systems in the healthy food domain. J Intell Inf Syst 50, 501–526 (2018)\
[https://doi.org/10.1007/s10844-017-0469-0](https://doi.org/10.1007/s10844-017-0469-0)

<a id="2">[2]</a>
Kaggle Data set for User food Interactions\
[https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv]()

<a id="3">[3]</a> 
A. -S. Metwalli, W. Shen and C. Q. Wu, "Food Image Recognition Based on Densely Connected Convolutional Neural Networks," 2020 International Conference on Artificial Intelligence in Information and Communication (ICAIIC), 2020, pp. 027-032\
[doi: 10.1109/ICAIIC48513.2020.9065281](https://www.researchgate.net/publication/340688231_Food_Image_Recognition_Based_on_Densely_Connected_Convolutional_Neural_Networks)

<a id="4">[4]</a>
Food 101 DataSet\
[https://www.kaggle.com/datasets/dansbecker/food-101](https://www.kaggle.com/datasets/dansbecker/food-101)

<a id="5">[5]</a>
Netflix Movie Reccomedation Competition 2006
[https://sifter.org/~simon/journal/20061211.html](https://sifter.org/~simon/journal/20061211.html)

<a id="6">[6]</a>
Surprise: A Python library for recommender systems, 2020
[https://doi.org/10.21105/joss.02174](https://doi.org/10.21105/joss.02174)

## Project Logistics
### Contribution table

| Sr. No. |            Stage Description            |       Contributors               |
|:-------:|:---------------------------------------:|:--------------------------------:|
|    1    |         Classification with CNN         |      Manoj + Anshit + Abhinav    |
|    2    | Querying Label with Ingredient Database |      Yibei + Ashish              |
|    3    |          Recommendation System          |     Abhinav + Ashish + Yibei     |
|    4    |                Deployment               |          Anshit                  |

*Subject to alterations*
### GANTT Chart

[GANTT Chart](https://gtvault-my.sharepoint.com/:x:/g/personal/averma373_gatech_edu/EVhkpnexSZlFo1E8W2ZUiFkBdDpVgO8g5v7mOKs5ekzM0Q?e=DuKFx6)
