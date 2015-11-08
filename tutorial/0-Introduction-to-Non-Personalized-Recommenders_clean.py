
# coding: utf-8

# # Introduction to Non-Personalized Recommenders

# ## About the authors
# 
# This IPython Notebook was forked from https://github.com/python-recsys/recsys-tutorial and modified by MSc Benjamin Tovar Cisneros (https://www.linkedin.com/in/benjamintovarcis/) on November 2015.
# 
# My forked version is available at https://github.com/TATABOX42/recsys-tutorial 
# 
# ## The recommendation problem
# 
# Recommenders have been around since at least 1992. Today we see different flavours of recommenders, deployed across different verticals: 
# 
# - Amazon
# - Netflix
# - Facebook
# - Last.fm.
# 
# What exactly do they do?
# 
# ### Definitions from the literature
# 
# *In a typical recommender system people provide recommendations as inputs, which
# the system then aggregates and directs to appropriate recipients.* -- Resnick
# and Varian, 1997
# 
# *Collaborative filtering simply means that people collaborate to help one
# another perform filtering by recording their reactions to documents they read.*
# -- Goldberg et al, 1992
# 
# *In its most common formulation, the recommendation problem is reduced to the
# problem of estimating ratings for the items that have not been seen by a
# user. Intuitively, this estimation is usually based on the ratings given by this
# user to other items and on some other information [...] Once we can estimate
# ratings for the yet unrated items, we can recommend to the user the item(s) with
# the highest estimated rating(s).* -- Adomavicius and Tuzhilin, 2005
# 
# *Driven by computer algorithms, recommenders help consumers
# by selecting products they will probably like and might buy
# based on their browsing, searches, purchases, and preferences.* -- Konstan and Riedl, 2012

# ### Notation
# 
# - $U$ is the set of users in our domain. Its size is $|U|$.
# - $I$ is the set of items in our domain. Its size is $|I|$.
# - $I(u)$ is the set of items that user $u$ has rated.
# - $-I(u)$ is the complement of $I(u)$ i.e., the set of items not yet seen by user $u$.
# - $U(i)$ is the set of users that have rated item $i$.
# - $-U(i)$ is the complement of $U(i)$.

# ### Goal of a recommendation system
# 
# $$ 
# \newcommand{\argmax}{\mathop{\rm argmax}\nolimits}
# \forall{u \in U},\; i^* = \argmax_{i \in -I(u)} [S(u,i)] 
# $$

# ### Problem statement
# 
# The recommendation problem in its most basic form is quite simple to define:
# 
# ```
# |-------------------+-----+-----+-----+-----+-----|
# | user_id, movie_id | m_1 | m_2 | m_3 | m_4 | m_5 |
# |-------------------+-----+-----+-----+-----+-----|
# | u_1               | ?   | ?   | 4   | ?   | 1   |
# |-------------------+-----+-----+-----+-----+-----|
# | u_2               | 3   | ?   | ?   | 2   | 2   |
# |-------------------+-----+-----+-----+-----+-----|
# | u_3               | 3   | ?   | ?   | ?   | ?   |
# |-------------------+-----+-----+-----+-----+-----|
# | u_4               | ?   | 1   | 2   | 1   | 1   |
# |-------------------+-----+-----+-----+-----+-----|
# | u_5               | ?   | ?   | ?   | ?   | ?   |
# |-------------------+-----+-----+-----+-----+-----|
# | u_6               | 2   | ?   | 2   | ?   | ?   |
# |-------------------+-----+-----+-----+-----+-----|
# | u_7               | ?   | ?   | ?   | ?   | ?   |
# |-------------------+-----+-----+-----+-----+-----|
# | u_8               | 3   | 1   | 5   | ?   | ?   |
# |-------------------+-----+-----+-----+-----+-----|
# | u_9               | ?   | ?   | ?   | ?   | 2   |
# |-------------------+-----+-----+-----+-----+-----|
# ```
# 
# *Given a partially filled matrix of ratings ($|U|x|I|$), estimate the missing values.*
# 

# ### Challenges
# 
# #### Availability of item metadata
# 
# Content-based techniques are limited by the amount of metadata that is available
# to describe an item. There are domains in which feature extraction methods are
# expensive or time consuming, e.g., processing multimedia data such as graphics,
# audio/video streams. In the context of grocery items for example, it's often the
# case that item information is only partial or completely missing. Examples
# include:
# 
# - Ingredients
# - Nutrition facts
# - Brand
# - Description
# - County of origin
# 
# #### New user problem
# 
# A user has to have rated a sufficient number of items before a recommender
# system can have a good idea of what their preferences are. In a content-based
# system, the aggregation function needs ratings to aggregate.
# 
# #### New item problem
# 
# Collaborative filters rely on an item being rated by many users to compute
# aggregates of those ratings. Think of this as the exact counterpart of the new
# user problem for content-based systems.
# 
# #### Data sparsity
# 
# When looking at the more general versions of content-based and collaborative
# systems, the success of the recommender system depends on the availability of a
# critical mass of user/item iteractions. We get a first glance at the data
# sparsity problem by quantifying the ratio of existing ratings vs $|U|x|I|$. A
# highly sparse matrix of interactions makes it difficult to compute similarities
# between users and items. As an example, for a user whose tastes are unusual
# compared to the rest of the population, there will not be any other users who
# are particularly similar, leading to poor recommendations.
# 

# ### Flow chart: the big picture
# 

# In[ ]:

from IPython.core.display import Image 
Image(filename='./imgs/recsys_arch.png')


# # The CourseTalk dataset: loading and first look
# 
# Loading of the CourseTalk database.
# 
# The CourseTalk data is spread across three files. Using the `pd.read_table`
# method we load each file:
# 

# ### Loading  the users dataset 

# In[ ]:

Image(filename='./imgs/example.png')


# In[ ]:

import pandas as pd

unames = ['user_id', 'username']
users = pd.read_table('./data/users_set.dat',
                      sep='|', header=None, names=unames)
# show output
users.head()


# ### Loading  the courses dataset 

# In[ ]:

mnames = ['course_id', 'title', 'avg_rating', 'workload', 'university', 'difficulty', 'provider']
courses = pd.read_table('./data/cursos.dat',
                       sep='|', header=None, names=mnames)
# show output
courses.head()


# ### Loading  the ratings dataset 

# In[ ]:

rnames = ['user_id', 'course_id', 'rating']
ratings = pd.read_table('./data/ratings.dat',
                        sep='|', header=None, names=rnames)
# show output
ratings.head()


# ## Using `pd.merge` we get it all into  one big DataFrame.

# In[ ]:

coursetalk = pd.merge(pd.merge(ratings, courses), users)
# show output
coursetalk.head()


# ## Extracting a subset of the original data frame

# In[ ]:

# set features
features = ["title","university","rating","avg_rating","difficulty"]
# Specific dataset
data2 = pd.DataFrame(coursetalk, columns = features)
# show output
data2.head()


# ### Using `groupby` function

# In[ ]:

bydifficulty = data2.groupby('difficulty')
bydifficulty.head(n=2)


# ## Dropping rows with `NULL` values

# In[ ]:

bydifficulty = data2.dropna().groupby('difficulty')
bydifficulty.head(n=2)


# ## Using `describe` function

# In[ ]:

bydifficulty['rating'].describe()


# ### Run another similar example

# In[ ]:

byuniversity = data2.dropna().groupby('university')
byuniversity.head(n=2)


# In[ ]:

byuniversity['rating'].describe()


# ## Running all together: drop `NULL` values and `groupby` University and course difficulty

# In[ ]:

byuniversity_and_difficulty = data2.dropna().groupby(['university',"difficulty"])
byuniversity_and_difficulty.head(n=2)


# In[ ]:

# using describe function
byuniversity_and_difficulty['rating'].describe()


# In[ ]:

# or we can use another function, such as mean
byuniversity_and_difficulty["rating"].mean()


# In[ ]:

# or we can just count the number of grouped events
byuniversity_and_difficulty["difficulty"].count()


# ### Allow ploting in notebook

# In[ ]:

get_ipython().magic(u'pylab inline')


# ### Plot the histogram of ratings

# In[ ]:

data2.rating.hist()


# # Collaborative filtering: generalizations of the aggregation function
# 
# ## Non-personalized recommendations
# 

# ### Groupby
# 
# The idea of groupby is that of *split-apply-combine*:
# 
# - split data in an object according to a given key;
# - apply a function to each subset;
# - combine results into a new object.

#  To get mean course ratings grouped by the provider, we can use the pivot_table method:

# In[ ]:

mean_ratings = coursetalk.pivot_table('rating','provider',aggfunc='mean')
mean_ratings = mean_ratings.sort_values(ascending=False)
pd.DataFrame(mean_ratings)


# Now let's filter down to courses that received at least 20 ratings (a completely arbitrary number);
# To do this, I group the data by course_id and use size() to get a Series of group sizes for each title:

# In[ ]:

ratings_by_title = coursetalk.groupby('title').size()
pd.DataFrame(ratings_by_title,columns=["# of ratings"]).head(n=10)


# In[ ]:

# set rating threshold
rating_threshold = 20
active_titles = ratings_by_title.index[ratings_by_title >= rating_threshold]
pd.DataFrame(active_titles).head()


# The index of titles receiving at least 20 ratings can then be used to select rows from mean_ratings above:
# 

# In[ ]:

# compute mean ratings for selected courses only
mean_ratings = coursetalk.pivot_table('rating','title', aggfunc='mean')
pd.DataFrame(mean_ratings)


# By computing the mean rating for each course, we will order with the highest rating listed first.
# 

# In[ ]:

# rank by avg rating
pd.DataFrame(mean_ratings.ix[active_titles].sort_values(ascending=False))


# To see the top courses among Coursera students, we can sort by the 'Coursera' column in descending order:
# 

# In[ ]:

mean_ratings = coursetalk.pivot_table('rating', # populate table with 
                                      'title', # set rows
                                      'provider',# set columns 
                                       aggfunc='mean' # define aggregator function (to be applied on 1st parameter)
                                     )
mean_ratings[0:10]


# In[ ]:

# get mean rating of course 6.00x: Introduction to Computer Science and Programming offered by MIT's EDX
mean_ratings['edx']["6.00x: Introduction to Computer Science and Programming"]


# In[ ]:

# show a data frame of all coursera active titles (>= 20 ratings) ordered by average ratings
pd.DataFrame(mean_ratings['coursera'][active_titles].dropna().sort_values(ascending=False))


# 
# Now, let's go further!  How about rank the courses with the highest percentage of ratings that are 4 or higher ?  % of ratings 4+
# 

# Let's start with a simple pivoting example that does not involve any aggregation. We can extract a ratings matrix as follows:
# 

# ## Transform the ratings frame into a ratings matrix

# In[ ]:

# transform the ratings frame into a ratings matrix
ratings_mtx_df = coursetalk.pivot_table('rating',
                                        'user_id',
                                        'title')
# print first 10 rows and columns
ratings_mtx_df.ix[ratings_mtx_df.index[0:10], ratings_mtx_df.columns[0:10]]


# Let's extract only the rating that are 4 or higher.

# In[ ]:

ratings_gte_4 = ratings_mtx_df[ratings_mtx_df>=4.0]
# with an integer axis index only label-based indexing is possible
ratings_gte_4.ix[ratings_gte_4.index[0:10], ratings_gte_4.columns[0:10]]


# Now picking the number of total ratings for each course and the count of ratings 4+ , we can merge them into one DataFrame.

# In[ ]:

ratings_gte_4_pd = pd.DataFrame({'total': ratings_mtx_df.count(), 'gte_4': ratings_gte_4.count()})
ratings_gte_4_pd.head(10)


# In[ ]:

# add ratio column
ratings_gte_4_pd['gte_4_ratio'] = (ratings_gte_4_pd['gte_4'] * 1.0)/ ratings_gte_4_pd.total
ratings_gte_4_pd.head(10)


# Let's now go easy. Let's count the number of ratings for each course, and order with the most number of ratings.
# 

# In[ ]:

ratings_by_title = coursetalk.groupby('title').size()
ratings_by_title = ratings_by_title.sort_values(ascending=False)
pd.DataFrame(ratings_by_title,columns=["# of enrolled students"]).head(n=10)


# Finally using the formula above that we learned, let's find out what the courses that most often occur wit the popular MOOC An introduction to Interactive Programming with Python by using the method "x + y/ x" .  For each course, calculate the percentage of Programming with python raters who also rated that course. Order with the highest percentage first, and voilá we have the top 5 moocs.

# In[ ]:

course_users = coursetalk.pivot_table('rating', 'title', 'user_id')
course_users.ix[course_users.index[0:10], course_users.columns[0:10]]


# First, let's get only the users that rated the course An Introduction to Interactive Programming in Python

# In[ ]:

ratings_by_course = coursetalk[coursetalk.title == 'An Introduction to Interactive Programming in Python']
ratings_by_course.set_index('user_id', inplace=True) # if False, user_id will start with 0s
# show output
ratings_by_course.head()


# Now, for all other courses let's filter out only the ratings from users that  rated the Python course.

# In[ ]:

their_ids = ratings_by_course.index
their_ratings = course_users[their_ids]
their_ratings[20:40]


# In[ ]:

course_users[their_ids].ix[course_users[their_ids].index[0:10], course_users[their_ids].columns[0:10]]


# By applying the division: number of ratings who rated Python Course and the given course / total of ratings who rated the Python Course we have  our percentage.

# In[ ]:

course_count =  their_ratings.ix['An Introduction to Interactive Programming in Python'].count()
course_count


# In[ ]:

select_user_id = 6
selected_user_id_df = pd.DataFrame(their_ratings[select_user_id]).dropna()
selected_user_id_df


# In[ ]:

n_courses_by_user = their_ratings.apply(lambda n_courses_by_user: n_courses_by_user.count(),axis=0)
pd.DataFrame(n_courses_by_user,columns=["# courses taken for each user"]).head(n=10)


# In[ ]:

n_users_enrolled = their_ratings.apply(lambda n_courses_by_user: n_courses_by_user.count(),axis=1)
# important: add pseudocount
n_users_enrolled = n_users_enrolled + 1
course_count = course_count + 1
# sort courses by number of users enrolled
n_users_enrolled = n_users_enrolled.sort_values(ascending=False)
# print output
pd.DataFrame(n_users_enrolled,columns=["# students enrolled"]).head(n=10)


# Ordering by the score, highest first excepts the first one which contains the course itself.

# In[ ]:

# compute score (n students enrolled / total number of students)
score = n_users_enrolled / float(course_count)
recommended_titles = pd.DataFrame(score.sort_values(ascending=False),columns=["Score"])
recommended_titles.head(n=10)


# In[ ]:

# Extract score for a specific course
course_name = "Python"
pd.DataFrame(score.sort_values(ascending=False),index=[course_name],columns=["Score"])


# In[ ]:

# Extract score for a specific course
course_name = "Jazz Improvisation"
pd.DataFrame(score.sort_values(ascending=False),index=[course_name],columns=["Score"])

