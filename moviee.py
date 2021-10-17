#%%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Import movies.csv dataset
data_movies = pd.read_csv('movies.csv')
print(data_movies.shape)
data_movies.head()

#%%
# Import ratings.csv dataset
data_ratings = pd.read_csv('ratings.csv')
print(data_ratings.shape)
data_ratings.head()
#%%
#merging two dataframes "movies.csv" and "ratings.csv"
data = data_ratings.merge(data_movies, how='left', on='movieId',validate='m:1')
print(data.shape)
data
#%%
min_movie = 100   # movie has to have been rated over 100 times
min_user = 20 # user has to have rated at least 20 times

users = data.groupby('userId')['rating'].count()
users = users.loc[users > min_user].index.values
movies = data.groupby('movieId')['rating'].count()
movies = movies.loc[movies > min_movie].index.values
filtered = data.loc[data.userId.isin(users) & data.movieId.isin(movies)]

print('Unfiltered: ', data.shape[0])
print('Filtered: ', filtered.shape[0])
print('Kept {}% of data'.format(round(filtered.shape[0]/data.shape[0], 2)*100))
#%%
users_movies = filtered.pivot_table(index='userId', columns='movieId', values='rating')
print(users_movies.shape)
users_movies
#%%
mat_users_movies = users_movies.replace(np.nan, 0)
mat_users_movies = mat_users_movies.astype(int)
print(mat_users_movies.shape)
mat_users_movies
#%%
import numpy as np
X = mat_users_movies

from sklearn.decomposition import NMF
model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(X)
H = model.components_

print(W)
print(H)
#%%
print(pd.DataFrame(W@H))
#%%
def Frobeniusnorm(X):

    return np.sqrt(np.sum(np.sum((abs(X)**2))))

X=np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
Frobeniusnorm(X)
#%%
import random
def losevalues(X,prop=1/2):
    enleve=round(np.count_nonzero(X)*prop)
    Y=np.ravel(X)
    i=np.random.choice(np.where(Y!=0)[0],replace=False,size=enleve)
    Y[i]=0
    return Y.reshape(X.shape[0],X.shape[1]),i
X=np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
print(losevalues(X))
#%%
import numpy as np
X = mat_users_movies

from sklearn.decomposition import NMF
model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(X)
H = model.components_

print(W)
print(H)
print(X)
Frobeniusnorm(X-W@H)
#%%
def difference(X,Y,i):

    count=0
    for j in (i):
        if abs(np.ravel(X)[j]-np.ravel(Y)[j])<1:
            count+=1
    print(count/i.size)

def evaluation(X,prop,**args):
    Xl,i=losevalues(X,prop)
    from sklearn.decomposition import NMF
    model = NMF(args)
    W = model.fit_transform(Xl)
    H = model.components_
    Y=W@H
    difference(X,Y,i)

n_components=2, init='random', random_state=0
import numpy as np
X = np.array(mat_users_movies)
evaluation(X,0.1)

# %%

# %%

