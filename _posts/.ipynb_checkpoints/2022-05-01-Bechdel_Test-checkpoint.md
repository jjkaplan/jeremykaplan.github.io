---
layout: post
title: "Bedchel Test."
subtitle: "Gender Equality in Movie Industry."

date: 2022-08-3 10:45:13 -0400
background: '/img/bank/background.png'

---

Gender equality has been one of the hot topics for long time. There have been numerous stucdies, activities, and even law changes in order order to improve the gender equalities. Altghouh there is visible development in this regard, There is still more to do.

In this project I will analyze the gender equality or improvement in the movie industry. For that purpose, Bechdel Test will be employed. The Bechdel Test is comming from the idea that Alison Bechdel introduced in a comic strip  and it has simple rules:
1. At least two women
2. The women need to talk to each other
3. They need to talk to each other about something other than a man  

<img src="https://images.unsplash.com/photo-1603202662706-62ead3176b8f?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=669&q=80" width="500" height="200" align="center"/>


The requirement of Bechdel Test is uncomplicated and asumming the majority of movies will be succesfull in terms of gender equality. 

### Looked for Couple of Insight 

  1. Any Change of woman employment in movies over the years.<br/>
  2. The relationship of movie budget and woman employment if any.<br/>
  3. Female director imapct on the woman employment rate.<br/>
  4. Movie budget imapct on the woman employment rate.<br/>
  5. The woman employment rate impact on the movie revenue imapact.


  


```python
# imports library
import urllib, json
import pandas as pd

```


```python
# Importing Data From bechdeltest.com by their API
df = pd.read_json('http://bechdeltest.com/api/v1/getAllMovies')

```


```python
df.head()

```

<iframe src = "/img/bechdel/table.html" height = "350px" width = "100%"></iframe>


There are five features in data frame. Rating is the most I will work on because here it will be the main criteria to check if the movie pass the test or not. Rating less than 3 means failre while greater than 3 means pass.

Since the world does not have such a succes on gender equality in past let`s analyze data after 1970s. 


```python
df['rating'].value_counts()
```
```python
data_bechdel = df[df['year']>=1970]
data_bechdel.head()
```





  <div id="df-bba25795-b78e-40b5-b8d9-a2d8c58c7502">
    <div class="colab-df-container">
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
      <th>imdbid</th>
      <th>year</th>
      <th>id</th>
      <th>rating</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1356</th>
      <td>0065531</td>
      <td>1970</td>
      <td>255</td>
      <td>0</td>
      <td>Le Cercle Rouge</td>
    </tr>
    <tr>
      <th>1357</th>
      <td>0065466</td>
      <td>1970</td>
      <td>583</td>
      <td>3</td>
      <td>Beyond the Valley of the Dolls</td>
    </tr>
    <tr>
      <th>1358</th>
      <td>0065421</td>
      <td>1970</td>
      <td>1122</td>
      <td>3</td>
      <td>AristoCats, The</td>
    </tr>
    <tr>
      <th>1359</th>
      <td>0066327</td>
      <td>1970</td>
      <td>1726</td>
      <td>1</td>
      <td>Santa Clause is Comin&amp;#39; to Town</td>
    </tr>
    <tr>
      <th>1360</th>
      <td>0064806</td>
      <td>1970</td>
      <td>1932</td>
      <td>1</td>
      <td>Phantom Tollbooth, The</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-bba25795-b78e-40b5-b8d9-a2d8c58c7502')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-bba25795-b78e-40b5-b8d9-a2d8c58c7502 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-bba25795-b78e-40b5-b8d9-a2d8c58c7502');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




I am now going to rename the column ‘rating’ to ‘Bechdel Score’, to make things clearer for the rest of the analysis.


```python
data_bechdel.rename(columns={'rating':'Bechdel Score'},inplace=True)
```

    /usr/local/lib/python3.7/dist-packages/pandas/core/frame.py:5047: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      errors=errors,



```python
data_bechdel.head()
```





  <div id="df-25b4d2ed-dab7-4484-9010-647af5a24d63">
    <div class="colab-df-container">
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
      <th>imdbid</th>
      <th>year</th>
      <th>id</th>
      <th>Bechdel Score</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1356</th>
      <td>0065531</td>
      <td>1970</td>
      <td>255</td>
      <td>0</td>
      <td>Le Cercle Rouge</td>
    </tr>
    <tr>
      <th>1357</th>
      <td>0065466</td>
      <td>1970</td>
      <td>583</td>
      <td>3</td>
      <td>Beyond the Valley of the Dolls</td>
    </tr>
    <tr>
      <th>1358</th>
      <td>0065421</td>
      <td>1970</td>
      <td>1122</td>
      <td>3</td>
      <td>AristoCats, The</td>
    </tr>
    <tr>
      <th>1359</th>
      <td>0066327</td>
      <td>1970</td>
      <td>1726</td>
      <td>1</td>
      <td>Santa Clause is Comin&amp;#39; to Town</td>
    </tr>
    <tr>
      <th>1360</th>
      <td>0064806</td>
      <td>1970</td>
      <td>1932</td>
      <td>1</td>
      <td>Phantom Tollbooth, The</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-25b4d2ed-dab7-4484-9010-647af5a24d63')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-25b4d2ed-dab7-4484-9010-647af5a24d63 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-25b4d2ed-dab7-4484-9010-647af5a24d63');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Here I am going to convert the ‘year’ column into a datetime object to analyze data. easily.


```python
data_bechdel['year'] = pd.to_datetime( data_bechdel['year'],format='%Y')
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.



```python
data_bechdel
```





  <div id="df-58dc379a-02e1-434a-b62f-b15f98da4117">
    <div class="colab-df-container">
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
      <th>imdbid</th>
      <th>year</th>
      <th>id</th>
      <th>Bechdel Score</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1356</th>
      <td>0065531</td>
      <td>1970-01-01</td>
      <td>255</td>
      <td>0</td>
      <td>Le Cercle Rouge</td>
    </tr>
    <tr>
      <th>1357</th>
      <td>0065466</td>
      <td>1970-01-01</td>
      <td>583</td>
      <td>3</td>
      <td>Beyond the Valley of the Dolls</td>
    </tr>
    <tr>
      <th>1358</th>
      <td>0065421</td>
      <td>1970-01-01</td>
      <td>1122</td>
      <td>3</td>
      <td>AristoCats, The</td>
    </tr>
    <tr>
      <th>1359</th>
      <td>0066327</td>
      <td>1970-01-01</td>
      <td>1726</td>
      <td>1</td>
      <td>Santa Clause is Comin&amp;#39; to Town</td>
    </tr>
    <tr>
      <th>1360</th>
      <td>0064806</td>
      <td>1970-01-01</td>
      <td>1932</td>
      <td>1</td>
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
      <td>13145534</td>
      <td>2022-01-01</td>
      <td>10370</td>
      <td>1</td>
      <td>Incroyable mais vrai</td>
    </tr>
    <tr>
      <th>9566</th>
      <td>10298810</td>
      <td>2022-01-01</td>
      <td>10371</td>
      <td>2</td>
      <td>Lightyear</td>
    </tr>
    <tr>
      <th>9567</th>
      <td>3513500</td>
      <td>2022-01-01</td>
      <td>10372</td>
      <td>3</td>
      <td>Chip &amp;#39;n Dale: Rescue Rangers</td>
    </tr>
    <tr>
      <th>9568</th>
      <td>14169960</td>
      <td>2022-01-01</td>
      <td>10375</td>
      <td>1</td>
      <td>All of Us Are Dead</td>
    </tr>
    <tr>
      <th>9569</th>
      <td>15521050</td>
      <td>2022-01-01</td>
      <td>10382</td>
      <td>3</td>
      <td>Love and Gelato</td>
    </tr>
  </tbody>
</table>
<p>8214 rows × 5 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-58dc379a-02e1-434a-b62f-b15f98da4117')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-58dc379a-02e1-434a-b62f-b15f98da4117 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-58dc379a-02e1-434a-b62f-b15f98da4117');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Next,  Bechdel Scores needs to be converted to categorical variables.


```python
data_bechdel['Bechdel Score'] = data_bechdel['Bechdel Score'].astype('category',copy=False)
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.



```python
data_bechdel.head()
```





  <div id="df-4fcfec71-d0f3-44a3-b9f1-2d15013bba80">
    <div class="colab-df-container">
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
      <th>imdbid</th>
      <th>year</th>
      <th>id</th>
      <th>Bechdel Score</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1356</th>
      <td>0065531</td>
      <td>1970-01-01</td>
      <td>255</td>
      <td>0</td>
      <td>Le Cercle Rouge</td>
    </tr>
    <tr>
      <th>1357</th>
      <td>0065466</td>
      <td>1970-01-01</td>
      <td>583</td>
      <td>3</td>
      <td>Beyond the Valley of the Dolls</td>
    </tr>
    <tr>
      <th>1358</th>
      <td>0065421</td>
      <td>1970-01-01</td>
      <td>1122</td>
      <td>3</td>
      <td>AristoCats, The</td>
    </tr>
    <tr>
      <th>1359</th>
      <td>0066327</td>
      <td>1970-01-01</td>
      <td>1726</td>
      <td>1</td>
      <td>Santa Clause is Comin&amp;#39; to Town</td>
    </tr>
    <tr>
      <th>1360</th>
      <td>0064806</td>
      <td>1970-01-01</td>
      <td>1932</td>
      <td>1</td>
      <td>Phantom Tollbooth, The</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-4fcfec71-d0f3-44a3-b9f1-2d15013bba80')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-4fcfec71-d0f3-44a3-b9f1-2d15013bba80 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-4fcfec71-d0f3-44a3-b9f1-2d15013bba80');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
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
     0   imdbid         8214 non-null   object        
     1   year           8214 non-null   datetime64[ns]
     2   id             8214 non-null   int64         
     3   Bechdel Score  8214 non-null   category      
     4   title          8214 non-null   object        
    dtypes: category(1), datetime64[ns](1), int64(1), object(2)
    memory usage: 329.1+ KB



```python

from matplotlib.pyplot import figure
import seaborn as sns
figure(figsize=(9,7))
ax = sns.countplot(x='Bechdel Score',data= data_bechdel);
for f in p.patches:
    ax.annotate('{:.1f}'.format(f.get_height()), (f.get_x()+0.3, f.get_height()+40))
plt.savefig('/img/bechdel/bchdel_score.png')
```
![png](Bechdel_Test_files/Bechdel_Test_18_0.png)
    



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

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      if __name__ == '__main__':





    1    4895
    0    3319
    Name: pass, dtype: int64




```python
figure(figsize=(9,7))
ax = sns.countplot(data=data_bechdel,x='pass')
for p in ax.patches:
   ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+40))

```



![png](Bechdel_Test_files/Bechdel_Test_20_0.png)
    



```python
from plotnine import *
(ggplot(data_bechdel)+geom_point(aes('year',color=data_bechdel['Bechdel Score']),
stat='count',show_legend=False)+geom_line(aes('year',color=data_bechdel['Bechdel Score']),
stat='count',show_legend=False))
```

    /usr/local/lib/python3.7/dist-packages/plotnine/utils.py:1246: FutureWarning: is_categorical is deprecated and will be removed in a future version.  Use is_categorical_dtype instead
      if pdtypes.is_categorical(arr):




![png](Bechdel_Test_files/Bechdel_Test_21_1.png)
    





    <ggplot: (8742399079873)>



Lets Check the relationship between Imdb rating and bechdel scores


```python

```


```python
from pandas.core.reshape.merge import merge

url = r"https://raw.githubusercontent.com/Natassha/Bechdel-Test/master/movies.csv"
imdb = pd.read_csv(url,encoding='cp1252')
imdb.head()
imdbNew = imdb[['title','rating']]
imdbNew
```





  <div id="df-2fe19199-e3d8-4bef-beb8-35abca4ceaeb">
    <div class="colab-df-container">
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
      <th>title</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>$</td>
      <td>6.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>$1000 a Touchdown</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>$21 a Day Once a Month</td>
      <td>8.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>$40,000</td>
      <td>8.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>$50,000 Climax Show, The</td>
      <td>3.4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>58783</th>
      <td>tom thumb</td>
      <td>6.5</td>
    </tr>
    <tr>
      <th>58784</th>
      <td>www.XXX.com</td>
      <td>1.1</td>
    </tr>
    <tr>
      <th>58785</th>
      <td>www.hellssoapopera.com</td>
      <td>6.6</td>
    </tr>
    <tr>
      <th>58786</th>
      <td>xXx</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>58787</th>
      <td>xXx: State of the Union</td>
      <td>3.9</td>
    </tr>
  </tbody>
</table>
<p>58788 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-2fe19199-e3d8-4bef-beb8-35abca4ceaeb')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-2fe19199-e3d8-4bef-beb8-35abca4ceaeb button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-2fe19199-e3d8-4bef-beb8-35abca4ceaeb');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





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



![png](Bechdel_Test_files/Bechdel_Test_27_0.png)
    



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





  <div id="df-7535d39b-f66a-429a-97b4-62ba40edc46b">
    <div class="colab-df-container">
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
      <button class="colab-df-convert" onclick="convertToInteractive('df-7535d39b-f66a-429a-97b4-62ba40edc46b')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-7535d39b-f66a-429a-97b4-62ba40edc46b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7535d39b-f66a-429a-97b4-62ba40edc46b');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Visualizing the relationship


```python
import matplotlib.pyplot as plt
```


```python

fig, ax = plt.subplots(figsize=(9, 6))
p = sns.scatterplot(x = 'year', y = 'rating',hue='Bechdel Score', data = new,ax=ax)
p.set_xlabel('year', fontsize = 17)
p.set_ylabel('rating', fontsize = 17)
p.set_title('Comparison of Rating and Bechdel Score Over the Yars', fontsize = 14, color='red')
# sns.lineplot(x="year", y="rating", hue="Bechdel Score",data=new,ax = ax);
```




    Text(0.5, 1.0, 'Comparison of Rating and Bechdel Score Over the Yars')




    
![png](Bechdel_Test_files/Bechdel_Test_33_1.png)
    



```python
# Plot year against IMDB rating and Bechdel Score:
ggplot(new,aes(x='year',y='rating',color='Bechdel Score'))+ \
geom_point()+geom_smooth()+\
scale_y_continuous(name="imdb rating")+\
labs( colour='Bechdel Score' )
```

    /usr/local/lib/python3.7/dist-packages/plotnine/utils.py:1246: FutureWarning: is_categorical is deprecated and will be removed in a future version.  Use is_categorical_dtype instead
      if pdtypes.is_categorical(arr):
    /usr/local/lib/python3.7/dist-packages/plotnine/stats/smoothers.py:168: PlotnineWarning: Confidence intervals are not yet implementedfor lowess smoothings.
      "for lowess smoothings.", PlotnineWarning)
    /usr/local/lib/python3.7/dist-packages/plotnine/layer.py:452: PlotnineWarning: geom_point : Removed 5 rows containing missing values.
      self.data = self.geom.handle_na(self.data)




![png](Bechdel_Test_files/Bechdel_Test_34_1.png)
    





    <ggplot: (8742398527613)>



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

    /usr/local/lib/python3.7/dist-packages/pandas/core/frame.py:5047: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      errors=errors,






  <div id="df-ce544c3f-0178-4488-adf7-9d81499b29ad">
    <div class="colab-df-container">
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
      <th>imdbid</th>
      <th>year</th>
      <th>Bechdel Score</th>
      <th>title</th>
      <th>pass</th>
      <th>rating</th>
      <th>director</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>0067800</td>
      <td>1971-01-01</td>
      <td>1</td>
      <td>Straw Dogs</td>
      <td>0</td>
      <td>7.4</td>
      <td>Rod Lurie</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0067741</td>
      <td>1971-01-01</td>
      <td>1</td>
      <td>Shaft</td>
      <td>0</td>
      <td>6.5</td>
      <td>John Singleton</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0067741</td>
      <td>1971-01-01</td>
      <td>1</td>
      <td>Shaft</td>
      <td>0</td>
      <td>6.0</td>
      <td>John Singleton</td>
    </tr>
    <tr>
      <th>53</th>
      <td>0067128</td>
      <td>1971-01-01</td>
      <td>1</td>
      <td>Get Carter</td>
      <td>0</td>
      <td>7.4</td>
      <td>Stephen Kay</td>
    </tr>
    <tr>
      <th>54</th>
      <td>0067128</td>
      <td>1971-01-01</td>
      <td>1</td>
      <td>Get Carter</td>
      <td>0</td>
      <td>4.7</td>
      <td>Stephen Kay</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ce544c3f-0178-4488-adf7-9d81499b29ad')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ce544c3f-0178-4488-adf7-9d81499b29ad button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ce544c3f-0178-4488-adf7-9d81499b29ad');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




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

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:17: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy






  <div id="df-2beb636c-cfad-4104-8a01-806a55a18f39">
    <div class="colab-df-container">
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
      <th>imdbid</th>
      <th>year</th>
      <th>Bechdel Score</th>
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
      <td>0067800</td>
      <td>1971-01-01</td>
      <td>1</td>
      <td>Straw Dogs</td>
      <td>0</td>
      <td>7.4</td>
      <td>Rod Lurie</td>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0067741</td>
      <td>1971-01-01</td>
      <td>1</td>
      <td>Shaft</td>
      <td>0</td>
      <td>6.5</td>
      <td>John Singleton</td>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0067741</td>
      <td>1971-01-01</td>
      <td>1</td>
      <td>Shaft</td>
      <td>0</td>
      <td>6.0</td>
      <td>John Singleton</td>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53</th>
      <td>0067128</td>
      <td>1971-01-01</td>
      <td>1</td>
      <td>Get Carter</td>
      <td>0</td>
      <td>7.4</td>
      <td>Stephen Kay</td>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>54</th>
      <td>0067128</td>
      <td>1971-01-01</td>
      <td>1</td>
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
      <button class="colab-df-convert" onclick="convertToInteractive('df-2beb636c-cfad-4104-8a01-806a55a18f39')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-2beb636c-cfad-4104-8a01-806a55a18f39 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-2beb636c-cfad-4104-8a01-806a55a18f39');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
figure(figsize=(9,7))
ax = sns.countplot(x='gender',data= data_combined)
for p in ax.patches:
   ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+10))

```



![png](Bechdel_Test_files/Bechdel_Test_40_0.png)
    


Next, I will visualize the gender of the director with the Bechdel score, to see if movies with female directors have a higher score.


```python
sns.set(font_scale=1.3)
figure(figsize=(9,7))
ax = sns.countplot(x='Bechdel Score',hue='gender',data=data_combined)
ax.set_xlabel('Bechdel Score',fontsize=20)
ax.set_ylabel('gender',fontsize=20)
for p in ax.patches:
   ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+10))

```



![png](Bechdel_Test_files/Bechdel_Test_42_0.png)
    



```python
ggplot(aes(x = 'year', y = 'Bechdel Score',color='gender'), data = data_combined)+geom_point()
```

    /usr/local/lib/python3.7/dist-packages/plotnine/utils.py:1246: FutureWarning: is_categorical is deprecated and will be removed in a future version.  Use is_categorical_dtype instead
      if pdtypes.is_categorical(arr):




![png](Bechdel_Test_files/Bechdel_Test_43_1.png)
    





    <ggplot: (8742395543781)>



Next, I will take a look at the variable budget, to see if there is any kind of correlation between the budget of a movie and it’s Bechdel score.


```python
data_combined['budget']=latest['budget']
ggplot(aes(x='year', y='budget',color='Bechdel Score'),
       data = data_combined)+geom_point()+geom_smooth()
```

    /usr/local/lib/python3.7/dist-packages/plotnine/utils.py:1246: FutureWarning: is_categorical is deprecated and will be removed in a future version.  Use is_categorical_dtype instead
      if pdtypes.is_categorical(arr):
    /usr/local/lib/python3.7/dist-packages/plotnine/stats/smoothers.py:168: PlotnineWarning: Confidence intervals are not yet implementedfor lowess smoothings.
      "for lowess smoothings.", PlotnineWarning)




![png](Bechdel_Test_files/Bechdel_Test_45_1.png)
    





    <ggplot: (8742394674601)>



Now, I will visualize the relationship between budget and gender of the director:


```python
ggplot(aes(x = 'year', y = 'budget',color='gender'), data = data_combined)+geom_point()+geom_smooth()
```

    /usr/local/lib/python3.7/dist-packages/plotnine/utils.py:1246: FutureWarning: is_categorical is deprecated and will be removed in a future version.  Use is_categorical_dtype instead
      if pdtypes.is_categorical(arr):




![png](Bechdel_Test_files/Bechdel_Test_47_1.png)
    





    <ggplot: (8742393042017)>



Visualizing the relationship between a movie’s budget, Bechdel score, and gender of director:


```python
figure(figsize=(8,6))
sns.violinplot(x='Bechdel Score',y='budget',hue='Male',data=data_combined
               )
plt.tick_params(axis='both', which='major', labelsize=30)
```



![png](Bechdel_Test_files/Bechdel_Test_49_0.png)
    



```python
data_combined['gross'] = latest['gross']
# Movie grossing with Bechdel score and gender:
sns.violinplot(x='Bechdel Score',y='gross',hue='Male',data=data_combined)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f37f67140d0>




    
![png](Bechdel_Test_files/Bechdel_Test_50_1.png)
    



```python
# Movie grossing with year and gender:
figure(figsize=(8,6))
ggplot(aes(x = 'year', y = 'gross',color='gender'), data = data_combined)+geom_point()
```

    /usr/local/lib/python3.7/dist-packages/plotnine/utils.py:1246: FutureWarning: is_categorical is deprecated and will be removed in a future version.  Use is_categorical_dtype instead
      if pdtypes.is_categorical(arr):



    <Figure size 576x432 with 0 Axes>



    
![png](Bechdel_Test_files/Bechdel_Test_51_2.png)
    





    <ggplot: (8742395862093)>



<dv class >
And that’s it! I managed to visualize relationships and answer all the data questions I set out to find answers to.

Some interesting data findings include:

1. Movies that fail the Bechdel test tend to have higher IMDB ratings.<br/>
2. Over the years, there has been an increasing number of movies that pass the Bechdel test.<br/>
3. Movies with female directors have higher Bechdel scores.
4. There is no clear correlation between the budget or revenue of a movie and its Bechdel score.
Of course, this analysis was only done with the data I gathered from three places, and might not represent the entire population of movies out there.

Furthermore, there may have been other variables present that affected the outcome of this analysis, and it might be a good idea to experiment with data from a couple of other places before coming to a conclusion.

Finally, I would like to mention that the Bechdel test is not necessarily the best benchmark to measure female representation in movies. It does not take into consideration how well written a female character is, neither does it measure meaningful depth of character.

However, it is one of the most well-known metrics used to expose gender bias and is the only test we have this kind of data on.

<dv class />


```python
# !pip install https://github.com/aaren/notedown/tarball/master
```


```python
#export to Markdown with cell outputs
!notedown /content/notebook_here.ipynb --to markdown > md_with_outputs.md
```


```python

```
