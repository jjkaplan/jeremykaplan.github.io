---
layout: post
title: "Capstone Project 1"
subtitle: "The first step towards to Data World."
date: 2022-06-7 10:45:13 -0400
background: '/img/posts/01.jpg'

---


#  Data Exploration Capstone Project 

## Objectives

Here’s we’re going to solve: ***which boroughs of London have seen the greatest increase in housing prices, on average, over the last two decades?***


A borough is just a fancy word for district. You may be familiar with the five boroughs of New York… well, there are 32 boroughs within Greater London [(here's some info for the curious)](https://en.wikipedia.org/wiki/London_boroughs). Some of them are more desirable areas to live in, and the data will reflect that with a greater rise in housing prices.



### 1. Sourcing and Loading 




#### 1.1. Importing Libraries


```python
# Let's import the pandas, numpy libraries as pd, and np respectively. 
import pandas as pd
import numpy as np



# Load the pyplot collection of functions from matplotlib, as plt 
from matplotlib import pyplot as plt
```

#### 1.2.  Loading the data
Your data comes from the [London Datastore](https://data.london.gov.uk/): a free, open-source data-sharing portal for London-oriented datasets. 


```python
# First, make a variable called url_LondonHousePrices, and assign it the following link, enclosed in quotation-marks as a string:
# https://data.london.gov.uk/download/uk-house-price-index/70ac0766-8902-4eb5-aab5-01951aaed773/UK%20House%20price%20index.xls

url_LondonHousePrices = "https://data.london.gov.uk/download/uk-house-price-index/70ac0766-8902-4eb5-aab5-01951aaed773/UK%20House%20price%20index.xls"

# The dataset we're interested in contains the Average prices of the houses, and is actually on a particular sheet of the Excel file. 
# As a result, we need to specify the sheet name in the read_excel() method.
# Put this data into a variable called properties.  
properties = pd.read_excel(url_LondonHousePrices, sheet_name='Average price', index_col= None)

properties.head(10)
```



```python
Data_copy = properties
```


```python
# Use 3 decimal places in output display
pd.set_option("display.precision", 3)
```

### 2. Cleaning, transforming, and visualizing
This second stage is arguably the most important part of any Data Science project. The first thing to do is take a proper look at the data. Cleaning forms the majority of this stage, and can be done both before or after Transformation.

The end goal of data cleaning is to have tidy data. When data is tidy: 

1. Each variable has a column.
2. Each observation forms a row.

Keep the end goal in mind as you move through this process, every step will take you closer. 


**2.1. Exploring your data** 

Think about your pandas functions for checking out a dataframe. 


```python
Data_copy.columns
```




    Index(['Unnamed: 0', 'City of London', 'Barking & Dagenham', 'Barnet',
           'Bexley', 'Brent', 'Bromley', 'Camden', 'Croydon', 'Ealing', 'Enfield',
           'Greenwich', 'Hackney', 'Hammersmith & Fulham', 'Haringey', 'Harrow',
           'Havering', 'Hillingdon', 'Hounslow', 'Islington',
           'Kensington & Chelsea', 'Kingston upon Thames', 'Lambeth', 'Lewisham',
           'Merton', 'Newham', 'Redbridge', 'Richmond upon Thames', 'Southwark',
           'Sutton', 'Tower Hamlets', 'Waltham Forest', 'Wandsworth',
           'Westminster', 'Unnamed: 34', 'Inner London', 'Outer London',
           'Unnamed: 37', 'NORTH EAST', 'NORTH WEST', 'YORKS & THE HUMBER',
           'EAST MIDLANDS', 'WEST MIDLANDS', 'EAST OF ENGLAND', 'LONDON',
           'SOUTH EAST', 'SOUTH WEST', 'Unnamed: 47', 'England'],
          dtype='object')




```python

```


```python

```


```python
Data_copy.duplicated().sum()
```




    0




```python

```


```python
Data_copy.isna().sum().sort_values(ascending=False)
```




    Unnamed: 37             321
    Unnamed: 47             321
    Unnamed: 34             321
    Outer London              0
    Richmond upon Thames      0
    Southwark                 0
    Sutton                    0
    Tower Hamlets             0
    Waltham Forest            0
    Wandsworth                0
    Westminster               0
    Inner London              0
    Unnamed: 0                0
    Newham                    0
    NORTH EAST                0
    NORTH WEST                0
    YORKS & THE HUMBER        0
    EAST MIDLANDS             0
    WEST MIDLANDS             0
    EAST OF ENGLAND           0
    LONDON                    0
    SOUTH EAST                0
    SOUTH WEST                0
    Redbridge                 0
    Merton                    0
    City of London            0
    Lewisham                  0
    Barking & Dagenham        0
    Barnet                    0
    Bexley                    0
    Brent                     0
    Bromley                   0
    Camden                    0
    Croydon                   0
    Ealing                    0
    Enfield                   0
    Greenwich                 0
    Hackney                   0
    Hammersmith & Fulham      0
    Haringey                  0
    Harrow                    0
    Havering                  0
    Hillingdon                0
    Hounslow                  0
    Islington                 0
    Kensington & Chelsea      0
    Kingston upon Thames      0
    Lambeth                   0
    England                   0
    dtype: int64




```python
# Displaying information for each column in DataFrame
Data_copy.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 321 entries, 1 to 321
    Data columns (total 49 columns):
     #   Column                Non-Null Count  Dtype         
    ---  ------                --------------  -----         
     0   Unnamed: 0            321 non-null    datetime64[ns]
     1   City of London        321 non-null    object        
     2   Barking & Dagenham    321 non-null    object        
     3   Barnet                321 non-null    object        
     4   Bexley                321 non-null    object        
     5   Brent                 321 non-null    object        
     6   Bromley               321 non-null    object        
     7   Camden                321 non-null    object        
     8   Croydon               321 non-null    object        
     9   Ealing                321 non-null    object        
     10  Enfield               321 non-null    object        
     11  Greenwich             321 non-null    object        
     12  Hackney               321 non-null    object        
     13  Hammersmith & Fulham  321 non-null    object        
     14  Haringey              321 non-null    object        
     15  Harrow                321 non-null    object        
     16  Havering              321 non-null    object        
     17  Hillingdon            321 non-null    object        
     18  Hounslow              321 non-null    object        
     19  Islington             321 non-null    object        
     20  Kensington & Chelsea  321 non-null    object        
     21  Kingston upon Thames  321 non-null    object        
     22  Lambeth               321 non-null    object        
     23  Lewisham              321 non-null    object        
     24  Merton                321 non-null    object        
     25  Newham                321 non-null    object        
     26  Redbridge             321 non-null    object        
     27  Richmond upon Thames  321 non-null    object        
     28  Southwark             321 non-null    object        
     29  Sutton                321 non-null    object        
     30  Tower Hamlets         321 non-null    object        
     31  Waltham Forest        321 non-null    object        
     32  Wandsworth            321 non-null    object        
     33  Westminster           321 non-null    object        
     34  Unnamed: 34           0 non-null      float64       
     35  Inner London          321 non-null    object        
     36  Outer London          321 non-null    object        
     37  Unnamed: 37           0 non-null      float64       
     38  NORTH EAST            321 non-null    object        
     39  NORTH WEST            321 non-null    object        
     40  YORKS & THE HUMBER    321 non-null    object        
     41  EAST MIDLANDS         321 non-null    object        
     42  WEST MIDLANDS         321 non-null    object        
     43  EAST OF ENGLAND       321 non-null    object        
     44  LONDON                321 non-null    object        
     45  SOUTH EAST            321 non-null    object        
     46  SOUTH WEST            321 non-null    object        
     47  Unnamed: 47           0 non-null      float64       
     48  England               321 non-null    object        
    dtypes: datetime64[ns](1), float64(3), object(45)
    memory usage: 125.4+ KB



```python
Data_copy.shape
```




    (321, 46)




```python
# Removing spaces in column names in sake of easy use
Data_copy.columns = Data_copy.columns.str.replace("\s+","_",regex=True)
```


**2.2. Cleaning the data**



```python

# Drop first row which makes hard to work on data and it does not such affect 

Data_copy.drop(0,inplace=True)
```

```python

# Rename date column
Data_copy.rename(columns={'Unnamed:_0':'Date'},inplace = True) 
```


```python
Data_copy
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
      <th>Date</th>
      <th>Barking_&amp;_Dagenham</th>
      <th>Barnet</th>
      <th>Bexley</th>
      <th>Brent</th>
      <th>Bromley</th>
      <th>Camden</th>
      <th>Croydon</th>
      <th>Ealing</th>
      <th>Enfield</th>
      <th>...</th>
      <th>Merton</th>
      <th>Newham</th>
      <th>Redbridge</th>
      <th>Richmond_upon_Thames</th>
      <th>Southwark</th>
      <th>Sutton</th>
      <th>Tower_Hamlets</th>
      <th>Waltham_Forest</th>
      <th>Wandsworth</th>
      <th>Westminster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1995-01-01</td>
      <td>50460.227</td>
      <td>93284.518</td>
      <td>64958.09</td>
      <td>71306.567</td>
      <td>81671.477</td>
      <td>120932.888</td>
      <td>69158.162</td>
      <td>79885.891</td>
      <td>72514.691</td>
      <td>...</td>
      <td>82070.613</td>
      <td>53539.319</td>
      <td>72189.584</td>
      <td>109326.125</td>
      <td>67885.203</td>
      <td>71536.974</td>
      <td>59865.19</td>
      <td>61319.449</td>
      <td>88559.044</td>
      <td>133025.277</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1995-02-01</td>
      <td>51085.78</td>
      <td>93190.17</td>
      <td>64787.921</td>
      <td>72022.262</td>
      <td>81657.559</td>
      <td>119508.862</td>
      <td>68951.095</td>
      <td>80897.066</td>
      <td>73155.197</td>
      <td>...</td>
      <td>79982.749</td>
      <td>53153.883</td>
      <td>72141.626</td>
      <td>111103.039</td>
      <td>64799.065</td>
      <td>70893.209</td>
      <td>62318.534</td>
      <td>60252.122</td>
      <td>88641.017</td>
      <td>131468.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1995-03-01</td>
      <td>51268.97</td>
      <td>92247.524</td>
      <td>64367.493</td>
      <td>72015.763</td>
      <td>81449.311</td>
      <td>120282.213</td>
      <td>68712.443</td>
      <td>81379.863</td>
      <td>72190.441</td>
      <td>...</td>
      <td>80661.683</td>
      <td>53458.264</td>
      <td>72501.355</td>
      <td>107325.474</td>
      <td>65763.297</td>
      <td>70306.838</td>
      <td>63938.677</td>
      <td>60871.085</td>
      <td>87124.815</td>
      <td>132260.342</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1995-04-01</td>
      <td>53133.505</td>
      <td>90762.875</td>
      <td>64277.669</td>
      <td>72965.631</td>
      <td>81124.412</td>
      <td>120097.899</td>
      <td>68610.046</td>
      <td>82188.905</td>
      <td>71442.922</td>
      <td>...</td>
      <td>79990.543</td>
      <td>54479.754</td>
      <td>72228.603</td>
      <td>106875</td>
      <td>63073.621</td>
      <td>69411.944</td>
      <td>66233.194</td>
      <td>60971.397</td>
      <td>87026.002</td>
      <td>133370.204</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1995-05-01</td>
      <td>53042.249</td>
      <td>90258.0</td>
      <td>63997.136</td>
      <td>73704.047</td>
      <td>81542.616</td>
      <td>119929.278</td>
      <td>68844.917</td>
      <td>82077.055</td>
      <td>70630.78</td>
      <td>...</td>
      <td>80873.986</td>
      <td>55803.96</td>
      <td>72366.641</td>
      <td>107707.68</td>
      <td>64420.499</td>
      <td>69759.22</td>
      <td>66432.858</td>
      <td>61494.169</td>
      <td>86518.059</td>
      <td>133911.112</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>317</th>
      <td>2021-05-01</td>
      <td>312430.653</td>
      <td>533319.35</td>
      <td>361919.178</td>
      <td>506917.307</td>
      <td>464872.548</td>
      <td>811080.363</td>
      <td>391236.913</td>
      <td>508368.502</td>
      <td>418235.579</td>
      <td>...</td>
      <td>540692.29</td>
      <td>380125.62</td>
      <td>451935.951</td>
      <td>701319.953</td>
      <td>507543.074</td>
      <td>395317.852</td>
      <td>438160.387</td>
      <td>476898.285</td>
      <td>600170.42</td>
      <td>908209.321</td>
    </tr>
    <tr>
      <th>318</th>
      <td>2021-06-01</td>
      <td>317234.735</td>
      <td>541296.783</td>
      <td>364464.187</td>
      <td>514055.899</td>
      <td>470431.417</td>
      <td>814571.246</td>
      <td>393198.407</td>
      <td>507915.956</td>
      <td>419803.094</td>
      <td>...</td>
      <td>541278.331</td>
      <td>382971.011</td>
      <td>455605.548</td>
      <td>698800.896</td>
      <td>507753.747</td>
      <td>396832.698</td>
      <td>441463.011</td>
      <td>474203.82</td>
      <td>602200.028</td>
      <td>888342.854</td>
    </tr>
    <tr>
      <th>319</th>
      <td>2021-07-01</td>
      <td>319549.93</td>
      <td>536012.669</td>
      <td>366451.639</td>
      <td>528684.055</td>
      <td>466447.004</td>
      <td>906099.2</td>
      <td>389730.086</td>
      <td>515141.329</td>
      <td>421158.795</td>
      <td>...</td>
      <td>544610.4</td>
      <td>366701.669</td>
      <td>461489.598</td>
      <td>719159.046</td>
      <td>519188.469</td>
      <td>401044.143</td>
      <td>459966.726</td>
      <td>493007.748</td>
      <td>602540.242</td>
      <td>922331.022</td>
    </tr>
    <tr>
      <th>320</th>
      <td>2021-08-01</td>
      <td>322496.375</td>
      <td>544558.807</td>
      <td>373547.295</td>
      <td>533696.39</td>
      <td>473570.356</td>
      <td>921525.014</td>
      <td>393245.781</td>
      <td>515422.283</td>
      <td>433986.209</td>
      <td>...</td>
      <td>559114.797</td>
      <td>379482.69</td>
      <td>470113.184</td>
      <td>736113.028</td>
      <td>534447.284</td>
      <td>401378.723</td>
      <td>466570.526</td>
      <td>502104.81</td>
      <td>614932.295</td>
      <td>1016725.048</td>
    </tr>
    <tr>
      <th>321</th>
      <td>2021-09-01</td>
      <td>329287.292</td>
      <td>533093.293</td>
      <td>375156.155</td>
      <td>542095.334</td>
      <td>483955.024</td>
      <td>895902.435</td>
      <td>391878.875</td>
      <td>525536.486</td>
      <td>440813.079</td>
      <td>...</td>
      <td>555225.54</td>
      <td>365357.143</td>
      <td>480339.944</td>
      <td>735675.219</td>
      <td>536755.525</td>
      <td>415227.665</td>
      <td>440846.736</td>
      <td>505389.186</td>
      <td>599124.413</td>
      <td>965766.355</td>
    </tr>
  </tbody>
</table>
<p>321 rows × 33 columns</p>
</div>




```python
# Added a month column to dataframe so that processing data related to month will be easy
Data_copy['month'] = pd.DatetimeIndex(Data_copy['Date']).month

m = Data_copy.pop('month')

Data_copy.insert(1,'month',m)
```


```python
# Added a year column to dataframe so that processing data related to year will be easy

Data_copy['Year'] = pd.DatetimeIndex(Data_copy['Date']).year
y = Data_copy.pop('Year')
y
Data_copy.insert(1,'Year',y)
```


```python
Data_copy.head()
```

**2.4.Transforming the data**
 
You might need to **melt** your DataFrame here. 


```python
# Reshape dataframe to work on data easier
Data_copy
mdata = pd.melt(Data_copy,id_vars=['Date','Year','month'],var_name="London_Boroughs",value_name="Price")

mdata.head()

```

we make sure your column data types are all correct. Average prices, for example, should be floating point numbers... 


```python
# price data type chamge to folat
mdata['Price'] = mdata['Price'].astype('float')
```

```python
# # Deleting non-related columns in Dataframe
# Since there only 32 London Boroughs, the rest needs to be removed as follow
Data_copy.drop(['Inner_London', 'Outer_London', 'NORTH_EAST',
       'NORTH_WEST', 'YORKS_&_THE_HUMBER', 'EAST_MIDLANDS', 'WEST_MIDLANDS',
       'EAST_OF_ENGLAND', 'LONDON', 'SOUTH_EAST', 'SOUTH_WEST', 'City_of_London','England'], axis=1,inplace=True)

# Deleting columns whoes all values are NaN
Data_copy.drop(['Unnamed:_34','Unnamed:_37', 'Unnamed:_47'],axis=1,inplace=True)

```
```python
mdata[mdata.Price>1000000].head(144)
```
**2.6. Visualizing the data**

```python
import seaborn as sns
import matplotlib_inline

```


```python
# Plotting Lonodn Boroughs house prices changes over years

plt.figure(figsize=(22,16))
sns.set(font_scale=1.3)

sns.scatterplot(mdata.Date,mdata.Price,data=mdata, hue ='London_Boroughs')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

# Set x-axis label
plt.xlabel('Date (Year)', fontsize =21)
# Set y-axis label
plt.ylabel('Price (million)', fontsize =21)
plt.title("Lonodn Boroughs house prices changes over years",fontsize =21)
plt.savefig("lnd_br_price.png")
```

```python
# 
# Average house price over years for Kensington_&_Chelsea
mean = mdata[mdata.London_Boroughs =='Kensington_&_Chelsea'].Price.mean()

# Plottting Average house price channges over years for Kensington_&_Chelsea against to average price
sns.relplot(
    data= mdata[mdata.London_Boroughs =='Kensington_&_Chelsea'], kind="line",
    x="Date", y="Price",
aspect=2.4
).set(
    title="Kensington_&_Chelsea House Prices Against to Average Price", 
    ylabel='Price (million)',
    xlabel='Date (Year)')

plt.hlines(mean,9500,19521,label='Ave House Prices');
plt.savefig('Knes&chelsePRice.png')
```
**3. Modeling**

Consider creating a function that will calculate a ratio of house prices, comparing the price of a house in 2018 to the price in 1998.

Consider calling this function create_price_ratio.

You'd want this function to:
1. Take a filter of dfg, specifically where this filter constrains the London_Borough, as an argument. For example, one admissible argument should be: dfg[dfg['London_Borough']=='Camden'].
2. Get the Average Price for that Borough, for the years 1998 and 2018.
4. Calculate the ratio of the Average Price for 1998 divided by the Average Price for 2018.
5. Return that ratio.

Once you've written this function, you ultimately want to use it to iterate through all the unique London_Boroughs and work out the ratio capturing the difference of house prices between 1998 and 2018.

Bear in mind: you don't have to write a function like this if you don't want to. If you can solve the brief otherwise, then great! 



```python

Camden = mdata[mdata['London_Boroughs']=='Camden']

```


```python
# Calculating average price changes from 1998 to 2018

def create_price_ratio():
    Camden['1998']=np.where(Camden['Date'].between('1/1/1998','12/1/1998'),'start','end')
    mean_1998 = Camden[Camden['1998']=='start'].Price.mean()
    Camden['2018']=np.where(Camden['Date'].between('1/1/2018','12/1/2018'),'start','end')
    mean_2018 =Camden[Camden['2018']=='start'].Price.mean()
    
    return (mean_1998/mean_2018)*100

create_price_ratio()
```

```python

```
