---
layout: post
title: "Gender Equality in Movie Industry."
subtitle: "Bechdel Test."

date: 2023- 05-06
background: '/img/bechdel/back.png'

---

Gender equality has been one of the hot topics for long time. There have been numerous stucdies, activities, and even law changes in order order to improve the gender equalities. Altghouh there is visible development in this regard, There is still more to do.

In this project I will analyze the gender equality or improvement in the movie industry. For that purpose, Bechdel Test will be employed. The Bechdel Test is comming from the idea that Alison Bechdel introduced in a comic strip  and it has simple rules:
1. At least two women
2. The women need to talk to each other
3. They need to talk to each other about something other than a man  


<!-- ![img]("https://images.unsplash.com/photo-1603202662706-62ead3176b8f?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=669&q=80")
 -->

The requirement of Bechdel Test is uncomplicated and asumming the majority of movies will be succesfull in terms of gender equality. 

### Looked for Couple of Insight 

  1. Any Change of woman employment in movies over the years.
  2. The relationship of movie budget and woman employment if any.
  3. Female director imapct on the woman employment rate.
  4. Movie budget imapct on the woman employment rate.
  5. The woman employment rate impact on the movie revenue imapact.


  


```python
# Importing Data From bechdeltest.com by their API
df = pd.read_json('http://bechdeltest.com/api/v1/getAllMovies')

```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>imdbid</th>
      <th>year</th>
      <th>id</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3155794</td>
      <td>1874</td>
      <td>9602</td>
      <td>Passage de Venus</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>14495706</td>
      <td>1877</td>
      <td>9804</td>
      <td>La Rosace Magique</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2221420</td>
      <td>1878</td>
      <td>9603</td>
      <td>Sallie Gardner at a Gallop</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>12592084</td>
      <td>1878</td>
      <td>9806</td>
      <td>Le singe musicien</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>7816420</td>
      <td>1881</td>
      <td>9816</td>
      <td>Athlete Swinging a Pick</td>
    </tr>
  </tbody>
</table>
</div>



<!-- <iframe src = "/img/bechdel/table.html" height = "350px" width = "100%"></iframe> -->

There are five features in data frame. Rating is the most I will work on because here it will be the main criteria to check if the movie pass the test or not. Rating less than 3 means failre while greater than 3 means pass.

Since the world does not have such a succes on gender equality in past let`s analyze data after 1970s. 


```python
df['rating'].value_counts()
```




    3    5451
    1    2085
    0    1069
    2     965
    Name: rating, dtype: int64




```python
data_bechdel = df[df['year']>=1970]
data_bechdel.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>imdbid</th>
      <th>year</th>
      <th>id</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1356</th>
      <td>0</td>
      <td>0065531</td>
      <td>1970</td>
      <td>255</td>
      <td>Le Cercle Rouge</td>
    </tr>
    <tr>
      <th>1357</th>
      <td>3</td>
      <td>0065466</td>
      <td>1970</td>
      <td>583</td>
      <td>Beyond the Valley of the Dolls</td>
    </tr>
    <tr>
      <th>1358</th>
      <td>3</td>
      <td>0065421</td>
      <td>1970</td>
      <td>1122</td>
      <td>AristoCats, The</td>
    </tr>
    <tr>
      <th>1359</th>
      <td>1</td>
      <td>0066327</td>
      <td>1970</td>
      <td>1726</td>
      <td>Santa Clause is Comin&amp;#39; to Town</td>
    </tr>
    <tr>
      <th>1360</th>
      <td>1</td>
      <td>0064806</td>
      <td>1970</td>
      <td>1932</td>
      <td>Phantom Tollbooth, The</td>
    </tr>
  </tbody>
</table>
</div>



I am now going to rename the column ‘rating’ to ‘Bechdel Score’, to make things clearer for the rest of the analysis.


```python
data_bechdel.rename(columns={'rating':'Bechdel Score'},inplace=True)
```


```python
data_bechdel.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bechdel Score</th>
      <th>imdbid</th>
      <th>year</th>
      <th>id</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1356</th>
      <td>0</td>
      <td>0065531</td>
      <td>1970</td>
      <td>255</td>
      <td>Le Cercle Rouge</td>
    </tr>
    <tr>
      <th>1357</th>
      <td>3</td>
      <td>0065466</td>
      <td>1970</td>
      <td>583</td>
      <td>Beyond the Valley of the Dolls</td>
    </tr>
    <tr>
      <th>1358</th>
      <td>3</td>
      <td>0065421</td>
      <td>1970</td>
      <td>1122</td>
      <td>AristoCats, The</td>
    </tr>
    <tr>
      <th>1359</th>
      <td>1</td>
      <td>0066327</td>
      <td>1970</td>
      <td>1726</td>
      <td>Santa Clause is Comin&amp;#39; to Town</td>
    </tr>
    <tr>
      <th>1360</th>
      <td>1</td>
      <td>0064806</td>
      <td>1970</td>
      <td>1932</td>
      <td>Phantom Tollbooth, The</td>
    </tr>
  </tbody>
</table>
</div>



<!-- <iframe src = "/img/bechdel/table2. html" height = "350px" width = "100%"></iframe> -->

Here I am going to convert the ‘year’ column into a datetime object to analyze data. easily.


```python
data_bechdel['year'] = pd.to_datetime( data_bechdel['year'],format='%Y')
```


```python
data_bechdel
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bechdel Score</th>
      <th>imdbid</th>
      <th>year</th>
      <th>id</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1356</th>
      <td>0</td>
      <td>0065531</td>
      <td>1970-01-01</td>
      <td>255</td>
      <td>Le Cercle Rouge</td>
    </tr>
    <tr>
      <th>1357</th>
      <td>3</td>
      <td>0065466</td>
      <td>1970-01-01</td>
      <td>583</td>
      <td>Beyond the Valley of the Dolls</td>
    </tr>
    <tr>
      <th>1358</th>
      <td>3</td>
      <td>0065421</td>
      <td>1970-01-01</td>
      <td>1122</td>
      <td>AristoCats, The</td>
    </tr>
    <tr>
      <th>1359</th>
      <td>1</td>
      <td>0066327</td>
      <td>1970-01-01</td>
      <td>1726</td>
      <td>Santa Clause is Comin&amp;#39; to Town</td>
    </tr>
    <tr>
      <th>1360</th>
      <td>1</td>
      <td>0064806</td>
      <td>1970-01-01</td>
      <td>1932</td>
      <td>Phantom Tollbooth, The</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9565</th>
      <td>1</td>
      <td>13145534</td>
      <td>2022-01-01</td>
      <td>10370</td>
      <td>Incroyable mais vrai</td>
    </tr>
    <tr>
      <th>9566</th>
      <td>2</td>
      <td>10298810</td>
      <td>2022-01-01</td>
      <td>10371</td>
      <td>Lightyear</td>
    </tr>
    <tr>
      <th>9567</th>
      <td>3</td>
      <td>3513500</td>
      <td>2022-01-01</td>
      <td>10372</td>
      <td>Chip &amp;#39;n Dale: Rescue Rangers</td>
    </tr>
    <tr>
      <th>9568</th>
      <td>1</td>
      <td>14169960</td>
      <td>2022-01-01</td>
      <td>10375</td>
      <td>All of Us Are Dead</td>
    </tr>
    <tr>
      <th>9569</th>
      <td>3</td>
      <td>15521050</td>
      <td>2022-01-01</td>
      <td>10382</td>
      <td>Love and Gelato</td>
    </tr>
  </tbody>
</table>
<p>8214 rows × 5 columns</p>
</div>



Next,  Bechdel Scores needs to be converted to categorical variables.


```python
data_bechdel['Bechdel Score'] = data_bechdel['Bechdel Score'].astype('category',copy=False)
```


```python
data_bechdel.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bechdel Score</th>
      <th>imdbid</th>
      <th>year</th>
      <th>id</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1356</th>
      <td>0</td>
      <td>0065531</td>
      <td>1970-01-01</td>
      <td>255</td>
      <td>Le Cercle Rouge</td>
    </tr>
    <tr>
      <th>1357</th>
      <td>3</td>
      <td>0065466</td>
      <td>1970-01-01</td>
      <td>583</td>
      <td>Beyond the Valley of the Dolls</td>
    </tr>
    <tr>
      <th>1358</th>
      <td>3</td>
      <td>0065421</td>
      <td>1970-01-01</td>
      <td>1122</td>
      <td>AristoCats, The</td>
    </tr>
    <tr>
      <th>1359</th>
      <td>1</td>
      <td>0066327</td>
      <td>1970-01-01</td>
      <td>1726</td>
      <td>Santa Clause is Comin&amp;#39; to Town</td>
    </tr>
    <tr>
      <th>1360</th>
      <td>1</td>
      <td>0064806</td>
      <td>1970-01-01</td>
      <td>1932</td>
      <td>Phantom Tollbooth, The</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_bechdel.describe()
data_bechdel.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 8214 entries, 1356 to 9569
    Data columns (total 5 columns):
     #   Column         Non-Null Count  Dtype         
    ---  ------         --------------  -----         
     0   Bechdel Score  8214 non-null   category      
     1   imdbid         8214 non-null   object        
     2   year           8214 non-null   datetime64[ns]
     3   id             8214 non-null   int64         
     4   title          8214 non-null   object        
    dtypes: category(1), datetime64[ns](1), int64(1), object(2)
    memory usage: 329.1+ KB



```python

from matplotlib.pyplot import figure
import seaborn as sns
figure(figsize=(9,7))
ax = sns.countplot(x='Bechdel Score',data= data_bechdel);
for f in ax.patches:
    ax.annotate('{:.1f}'.format(f.get_height()), (f.get_x()+0.3, f.get_height()+40))


```


![png](/img/bechdel/Bechdel_Test_18_0.png)
    


```python
# Lets check if the movies pass bechdel test
li = []
for i in data_bechdel['Bechdel Score']:
  if(i<3):
    li.append(0)
  else:
    li.append(1)

data_bechdel['pass'] = li
data_bechdel.head()
data_bechdel['pass'].value_counts()
```




    1    4895
    0    3319
    Name: pass, dtype: int64




```python
figure(figsize=(9,7))
ax = sns.countplot(data=data_bechdel,x='pass')
for p in ax.patches:
   ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+40))

```



![png](/img/bechdel/Bechdel_Test_21_0.png)
    


<!-- <iframe src = "/img/bechdel/graph2.png" height = "350px" width = "100%"></iframe> -->

Lets Check the relationship between Imdb rating and bechdel scores


```python
from pandas.core.reshape.merge import merge

url = r"https://raw.githubusercontent.com/Natassha/Bechdel-Test/master/movies.csv"
imdb = pd.read_csv(url,encoding='cp1252')
imdb.head()
imdbNew = imdb[['title','rating']]
imdbNew
```



```python
(ggplot(data_bechdel)+geom_point(aes('year',color=data_bechdel['Bechdel Score']),stat='count',show_legend=False)+geom_line(aes('year',color=data_bechdel['Bechdel Score']),stat='count',show_legend=False))

```
![png](/img/bechdel/Bechdel_Test_25_0.png)
    





    <ggplot: (8762740129624)>



<!-- <iframe src = "/img/bechdel/table3.png" height = "350px" width = "100%"></iframe> -->


```python
data_bechdel = pd.merge(data_bechdel,imdbNew, how ='left',
                 left_on = ['title'], right_on = ['title'])
```


```python

data_bechdel.head()
data_bechdel['rating'].value_counts().sort_index(ascending=False)
```




    9.9    1
    9.8    3
    9.6    2
    9.5    1
    9.4    1
          ..
    1.9    2
    1.8    2
    1.7    1
    1.6    2
    1.2    1
    Name: rating, Length: 81, dtype: int64




```python
figure(figsize=(8,6))
sns.scatterplot(x =data_bechdel['Bechdel Score'],
                y = data_bechdel['rating'],data=data_bechdel);
```



![png](/img/bechdel/Bechdel_Test_29_0.png)
    


<!-- <iframe src = "/img/bechdel/graph4.png" height = "350px" width = "100%"></iframe> -->


```python
# Droping rows with null values
data_bechdel.dropna(inplace=True)
data_bechdel.drop('id',axis=1,inplace=True)
```

Creaing new dataframe with only year, bechdel score and rating


```python
new = data_bechdel.groupby(['year','Bechdel Score']).agg({'rating':'mean'}).reset_index()
new.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>Bechdel Score</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1970-01-01</td>
      <td>0</td>
      <td>7.150000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1970-01-01</td>
      <td>1</td>
      <td>7.054545</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1970-01-01</td>
      <td>2</td>
      <td>6.866667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1970-01-01</td>
      <td>3</td>
      <td>6.440000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1971-01-01</td>
      <td>0</td>
      <td>6.875000</td>
    </tr>
  </tbody>
</table>
</div>



<!-- <iframe src = "/img/bechdel/table4.html" height = "350px" width = "100%"></iframe> -->

## Visualizing the relationship


```python

fig, ax = plt.subplots(figsize=(9, 6))
p = sns.scatterplot(x = 'year', y = 'rating',hue='Bechdel Score', data = new,ax=ax)
p.set_xlabel('year', fontsize = 17)
p.set_ylabel('rating', fontsize = 17)
p.set_title('Comparison of Rating and Bechdel Score Over the Yars', fontsize = 14, color='red');
```
![png](/img/bechdel/Bechdel_Test_36_0.png)
    


<!-- <iframe src = "/img/bechdel/graph5.png" height = "350px" width = "100%"></iframe> -->


```python
# Plot year against IMDB rating and Bechdel Score:
ggplot(new,aes(x='year',y='rating',color='Bechdel Score'))+ geom_point()+geom_smooth()+scale_y_continuous(name="imdb rating")+labs( colour='Bechdel Score' )+ ggtitle("Bechdel Score vs IMDB Rating Changes Over the Years")
```
![png](/img/bechdel/Bechdel_Test_38_0.png)
    





    <ggplot: (8762740129783)>



<!-- <iframe src = "/img/bechdel/graph6.png" height = "350px" width = "100%"></iframe> -->

It appears as though movies that pass the Bechdel test have significantly lower IMDB ratings compared to movies that don’t, which was pretty surprising to me.

Now, I will try to visualize the relationship between the gender of the director and Bechdel scores. I assume that movies with female directors are more likely to have higher Bechdel scores, which I will try to plot here.


```python
url = r"https://raw.githubusercontent.com/Natassha/Bechdel-Test/master/movielatest.csv"
latest = pd.read_csv(url,encoding='cp1252')
latest.head()
dfLatest = latest[['name','director']]
dfLatest.rename(columns={'name':'title'}, inplace=True)
data_combined = pd.merge(data_bechdel, dfLatest, how='left', left_on=['title'], right_on=['title'])
data_combined = data_combined.dropna()
data_combined.head()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bechdel Score</th>
      <th>imdbid</th>
      <th>year</th>
      <th>title</th>
      <th>pass</th>
      <th>rating</th>
      <th>director</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>1</td>
      <td>0067800</td>
      <td>1971-01-01</td>
      <td>Straw Dogs</td>
      <td>0</td>
      <td>7.4</td>
      <td>Rod Lurie</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1</td>
      <td>0067741</td>
      <td>1971-01-01</td>
      <td>Shaft</td>
      <td>0</td>
      <td>6.5</td>
      <td>John Singleton</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1</td>
      <td>0067741</td>
      <td>1971-01-01</td>
      <td>Shaft</td>
      <td>0</td>
      <td>6.0</td>
      <td>John Singleton</td>
    </tr>
    <tr>
      <th>53</th>
      <td>1</td>
      <td>0067128</td>
      <td>1971-01-01</td>
      <td>Get Carter</td>
      <td>0</td>
      <td>7.4</td>
      <td>Stephen Kay</td>
    </tr>
    <tr>
      <th>54</th>
      <td>1</td>
      <td>0067128</td>
      <td>1971-01-01</td>
      <td>Get Carter</td>
      <td>0</td>
      <td>4.7</td>
      <td>Stephen Kay</td>
    </tr>
  </tbody>
</table>
</div>



<!-- <iframe src = "/img/bechdel/table5.html" height = "350px" width = "100%"></iframe> -->

The newly created data frame now has an additional variable in it; director. I will now try to predict the gender of the director given their first name, and append it to the data frame.


```python
# !pip install gender-guesser

```


```python

import gender_guesser.detector as gen
# Predicting gender of director from first name:
d = gen.Detector()
genders = []
firstNames = data_combined['director'].str.split().str.get(0)
for i in firstNames[0:len(firstNames)]:
    if d.get_gender(i) == 'male':
        genders.append('male')
    elif d.get_gender(i) == 'female':
        genders.append('female')
    else:
        genders.append('unknown')
data_combined['gender'] = genders
data_combined = data_combined[data_combined['gender'] != 'unknown']
# Encode the variable gender into a new dataframe:
data_combined['Male'] = data_combined['gender'].map( {'male':1, 'female':0} )
data_combined.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bechdel Score</th>
      <th>imdbid</th>
      <th>year</th>
      <th>title</th>
      <th>pass</th>
      <th>rating</th>
      <th>director</th>
      <th>gender</th>
      <th>Male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>1</td>
      <td>0067800</td>
      <td>1971-01-01</td>
      <td>Straw Dogs</td>
      <td>0</td>
      <td>7.4</td>
      <td>Rod Lurie</td>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1</td>
      <td>0067741</td>
      <td>1971-01-01</td>
      <td>Shaft</td>
      <td>0</td>
      <td>6.5</td>
      <td>John Singleton</td>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1</td>
      <td>0067741</td>
      <td>1971-01-01</td>
      <td>Shaft</td>
      <td>0</td>
      <td>6.0</td>
      <td>John Singleton</td>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53</th>
      <td>1</td>
      <td>0067128</td>
      <td>1971-01-01</td>
      <td>Get Carter</td>
      <td>0</td>
      <td>7.4</td>
      <td>Stephen Kay</td>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>54</th>
      <td>1</td>
      <td>0067128</td>
      <td>1971-01-01</td>
      <td>Get Carter</td>
      <td>0</td>
      <td>4.7</td>
      <td>Stephen Kay</td>
      <td>male</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



<!-- <iframe src = "/img/bechdel/table6.html" height = "350px" width = "100%"></iframe> -->


```python
figure(figsize=(9,7))
ax = sns.countplot(x='gender',data= data_combined)

plt.title("Number of Diredtors in Both Gender")
for p in ax.patches:
    ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+10))

```
![png](/img/bechdel/Bechdel_Test_47_0.png)
    


<!-- <iframe src = "/img/bechdel/graph.png7" height = "350px" width = "100%"></iframe> -->

Next, I will visualize the gender of the director with the Bechdel score, to see if movies with female directors have a higher score.


```python
sns.set(font_scale=1.3)
figure(figsize=(9,7))
ax = sns.countplot(x='Bechdel Score',hue='gender',data=data_combined)
ax.set_xlabel('Bechdel Score',fontsize=20)
ax.set_ylabel('gender',fontsize=20)
plt.title("Number of Each Score for Both Gender Directors")
for p in ax.patches:
    ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+10))

```
![png](/img/bechdel/Bechdel_Test_50_0.png)
    


<!-- <iframe src = "/img/bechdel/graph.png8" height = "350px" width = "100%"></iframe> -->

Next, I will take a look at the variable budget, to see if there is any kind of correlation between the budget of a movie and it’s Bechdel score.


```python
data_combined['budget']=latest['budget']
ggplot(data_combined,aes(x = 'year', y ='budget' ,color='Bechdel Score'))+geom_point()+geom_smooth()+ ggtitle("Bechdel Score vs Budget Changes Over the Years")

```



![png](/img/bechdel/Bechdel_Test_53_0.png)
    





    <ggplot: (8762740166012)>



<!-- <iframe src = "/img/bechdel/graph.png9" height = "350px" width = "100%"></iframe> -->

Now, I will visualize the relationship between budget and gender of the director:


```python
data_combined['budget']=latest['budget']
# Visualize budget and gender of director
ggplot(data_combined,aes(x = 'year', y = 'budget',color='gender'))+\
geom_point()+geom_smooth()+\
ggtitle("Bechdel Score vs Gender Changes Over the Years")

```
![png](/img/bechdel/Bechdel_Test_56_0.png)
    





    <ggplot: (8762622883904)>



<!-- <iframe src = "/img/bechdel/graph.png10" height = "350px" width = "100%"></iframe> -->


```python
data_combined['gross']=latest['gross']
ggplot(data_combined,aes(x = 'year', y = 'gross',color='gender'))+geom_point()+ ggtitle("Gross Revenue vs Director Gender Changes Over the Years")

```



![png](/img/bechdel/Bechdel_Test_58_0.png)
    





    <ggplot: (8762724703166)>



<!-- <iframe src = "/img/bechdel/graph.png11" height = "350px" width = "100%"></iframe> -->



From the above analysis, it can be observed that number of movies that pass the Bedchel test has been increased.
Another result is that female directors have positive impact on movies Bedchel test performance.
Although the Bedchel test is well-known test, it is not best one to make conclusion about the movies since there different parameters that need to be taken in consideration.


[Reference]('https://www.natasshaselvaraj.com/')



