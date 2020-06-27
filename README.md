# Snapchat Political Ads Project
### By: Ronaldo Romano and Jason Sheu

## Summary of Findings

### Introduction
<span style="color:Red">**The Investigative Questions**</span>
> The questions me and my partner are investigating are: what are the most prevalent organizations, advertisers, and ballot candidates in the data, and how does the spend differ within single gendered ads versus both gender ads?

<span style="color:Red">**The Prediction Question**</span>
> The classification prediction we are trying to figure out is if a company will either spend above or below the average amount of money spent on ads throughout the whole dataset. We decided to evaluate our model based on accuracy. This is because in our question, the true positives and true negatives are more crucial than the false positives and false negatives. Furthermore, our model would classify false positives and negatives as a “wrong” prediction, lowering our accuracy score, so in our case false positives and false negatives are the same, resulting in a symmetrical dataset. We decided to go with accuracy to get a good perspective on how accurate our model would be on our testing data. We used a K-Neighbors classifier to create our classification model.

<span style="color:Red">**Choice of Target Variable & Evaluation Metric**</span>
> As our target variable we chose ‘Spend’, whether it would be spend above the average or below the average. For both of our models we chose to do accuracy, so from the baseline to the final our goal was to increase the accuracy. 

### Cleaning and EDA
<span style="color:Red">**Basic Cleaning**</span>
> We first imported the 2018 and 2019 datasets from our raw csv files, and read them into dataframes. We then concatenated the two dataframes together into one master dataframe with all of the information from 2018 and 2019. We cleaned the StartDate and EndDate columns by converting them into DateTime timestamp objects to make it more accessible and malleable. 
> For the 'Gender' column we filled the nulls with 'BOTH' and filled all the 'MALE'/'FEMALE' rows with 'SINGLE' to show that they were geared to only a single gender. For the 'CountryCode' column we combined all countries that weren't 'united states' into a single category 'other', as the majority of the ads came from the U.S. The 'Spend' column is in different currencies, so we convert them all to USD using the appropriate conversion rates. In the 'Spend' column we decided to make it 0/1 to show whether or not the company spent over the average. With the 'StartDate' and 'EndDate' we decided to get the length the ad was running for.

<span style="color:Red">**EDA for "What are the most prevalent organizations, advertisers, and ballot candidates in the data?"**</span>
> Then we performed EDA on both of our topics. For the topic about organizer prevalence and spending, we first checked the missingness of the spending values and organizer names for each ad and found that there were none. We wanted to then explore the companies with the most ads bought, so we took the master data and took the value counts of the OrganizationName column which gave us the number of ads each company bought. We then plotted the frequency of each ad volume on a histogram in <span style="color:Red">**Plot #1**</span>. In this we found that most companies buy less than 50 ads total, and very few exceed 50 ads bought. We also explored the average amount of money spent on an ad in relation to how many ads were bought. We first selected a smaller dataframe with the organization name and how much they spent on each ad, then we grouped by the organization with mean() as the aggregate function, allowing us to see the average amount of money they spent per ad. We merged this dataframe with the previously made dataframe of the number of ads bought by each company to get a dataframe that displayed the average amount spent on ads and the number of ads bought. We then plotted a scatterplot comparing the two in <span style="color:Red">**Plot #2**</span>, letting us find that in general companies try to spend as little as possible, and don't usually buy that many ads. 

<span style="color:Red">**EDA for "How does the targeted gender/age bracket of the ad affect the amount of money spent on the ad?"**</span>
> So for the analysis on gender/age affecting spend, we first looked at missingness within these three columns. As mentioned above, spend does not have nulls and the age bracket/gender did have nulls, but it was for a reason (either all ages or both genders). For <span style="color:Red">**Plot #3**</span>, we take a look at the number of ads for gender and we see that the majority of these include both male and female. Around 2% is strictly for males and 7% for just females. In <span style="color:Red">**Plot #4**</span>, we wanted to see how many ads there were for each age group. For us, the AgeBracket column was a bit hard to work with, so we decided to go ahead and only use the rows that had an age followed with a ‘+’ and that accounted for about 58% of the dataset. We can see from the bar graph that most of the ads are tailored for those over 18 years of age, this seemed reasonable since those that are eligible to vote have to be 18 years of age. Advertisers wouldn’t want to spend money on younger people who can’t vote in elections. In <span style="color:Red">**Plot #5**</span>, we plot a pie chart and we do it with money spent depending on gender. These 2 different categories account for only ~9% of the data (385 rows), but after plotting it we can see that more money is spent on ads geared toward females. This might be a little more weighted towards women since there are more single gender ads geared towards females. In <span style="color:Red">**Plot #6**</span>, we plot a scatter plot using spend as well as age brackets. Once again we run into a similar problem as earlier where we only use the rows that have an age followed by a ‘+’. From the plot, we can see that the money spent on ads for 18+ is the highest. A reasoning behind this may be that most advertisers wouldn’t want to make different types of ads for a lot of specific age brackets, spending ads for a more general public, like 18+ makes it so that the ad reaches basically everyone. We also noticed that many wouldn’t want to spend money on 30+, 33+, 34+ and so on because those age groups aren’t really on Snapchat anymore, this money can be used somewhere else.

### Assessment of Missingness
The question we posed was: <span style="color:Red">**"Is the Candidate Ballot Information missing at random dependent on the amount of spend?"**</span>

> The reason we used Candidate Ballot Information was because the other columns we were using to answer the question had meaning behind their missingness, like 'Age' and 'Gender'. After calculating the p-value, we saw that it was above the 5% significance level we had, so in this case we would have to say that Candidate Ballot Information is dependent on spend. A reason that could explain this is that maybe the money spent on the Ad wasn't meant to help any specific party, it was just for election in general. Another reason we came up with is that maybe those Ads that didn't really cost as much and have Candidate Info tied to them will make them seem like a weaker candidate.

The second question we posed was: <span style="color:Red">**"Are the Regions (Included) missing at random dependent on the amount of spend?"**</span>
> The p-value we got after the permutation was lower than our 5% significance level meaning that Regions (Included) is not dependent on Spend. If an element in Regions (Included) happens to be missing then it is not because of the Spend but might be because of another column/factor that we did not explore. The Region of the Ad shouldn't determine the cost as an Ad generally costs very similar to other ads, or it would be influenced by other factors not explored in this missingness dependence test.

### Hypothesis Test
<span style="color:Red">**First Permutation Test**</span>
> - **Null Hypothesis:** The number of ads bought does not affect the average amount of money spent on ads. 
> - **Alternative Hypothesis:**  The number of ads bought does have an effect on the amount of money spent on ads. 
> - **Test Statistic:** Total Variation Distance, our observed statistic was 47.748947856403106
> - **Significance Level:** 5% 
> - **Resulting P-Value:** 0.4601, Fail to Reject Null Hypothesis
> - **Results:** After looking at the output, we see that we fail to reject the null hypothesis and allow us to say that the number of ads bought does not have any effect on the amount spent on each ad.  This is interesting because it shows that on average, even when a company buys a large amount of advertisements, the average cost they spend is about the same, which can be interpreted as a company being consistent in their investment in advertisements.  

<span style="color:Red">**Second Permutation Test**</span>
> - **Null Hypothesis:** The average amount of money spent would be the same regardless of whether the ad is targeted for both genders or just a single gender (male/female). 
> - **Alternative Hypothesis:**  The average amount of money spent is not the same whether the ad is targeted for both genders or just a single gender (male/female). 
> - **Test Statistic:** Difference in Means, our observed statistic was 1009.5322612854538. 
> - **Significance Level:** 5% 
> - **Resulting P-Value:** 0.0016, Reject Null Hypothesis
> - **Results:** After looking at the output, we see that we can reject the null and therefore say that the average amount of money is not the same whether the ad is for both or just a single gender. This allows us to say that Spend is dependent on Gender groups. There is bias here, since around 90% of the data is meant for both genders instead of just one gender. We do think that these were the best columns to use when answering the second question, it does yield an answer that is accetable. 

### Baseline Model
> To create our baseline model, we created and used 3 features. We used a one hot encoded gender and country column, as well as an ordinally encoded organization name column. All of our features are nominally encoded, as they are either ordinally encoded or one-hot encoded. We trained our model on these three features, then calculated the accuracy of our prediction. We did this by seeing how many times the prediction predicted correctly on the testing data. We then ran a prediction 1000 times to find the mean accuracy, which was around <span style="color:Red">**0.789**</span>. We considered this to be an alright baseline model but also felt that there could be room for improvement. We also felt that using naturally numerical numbers would yield a good improvement, which we tried in our final model. This showed that our model was accurate on average <span style="color:Red">**78.9%**</span> of the time, which was pretty good for our baseline. 

### Final Model
> After playing around with different features, we decided to completely change our final model. In our final model, we trained our pipeline around 3 different features: the number of Impressions the ad had, the length of the ad, and the number of impressions per hour the ad had. We felt that these features would greatly improve our model, because generally the number of impressions an ad has can be related to the amount spent. We figured that the more an organization spent on an ad, the more people they would reach. We also figured that a longer ad would cost more money and more likely to be above the average spent. We calculated the length of the ad by first subtracting the end and start time, then converting the days to hours. We calculated the impressions per hour an ad had by dividing the impressions by the total hours the ad had. We decided to use these three features because we tested the features individually and found that they individually had better accuracies than the previous features. After training our model, we calculated the accuracy of it by doing the same thing as the baseline model: seeing how many predictions our model got correct. After running it 1000 times and taking the average accuracy, our model had <span style="color:Red">**0.946**</span> accuracy on average, which is a <span style="color:Red">**0.157 increase**</span> in accuracy from our baseline, which we were very satisfied with. This meant that our model is now <span style="color:Red">**95.6%**</span> accurate on average. 

### Fairness Evaluation
> In our fairness evaluation, we decided to test to see if the precision of our model would be better or worse on 2 subsets of the original dataset: short ads and long ads. The reason we chose to use precision score was to see whether having a short or long ad influenced the model in predicting that a company with below average spend was actually spending above average spend. To do this, we calculated the observed difference in precision between short ads and long ads, which was a difference of 0.004777839618733903. To see if our model is biased towards either short ads or long ads, we performed a permutation test with a significance level of 0.05. Our null hypothesis is that our model is fair and the precision for our two subsets of data are roughly the same. Our alternative hypothesis is that our model is unfair, the precision of the long ad subset is higher than the short ad subset. After performing our permutation test 1000 times, we got a p-value of <span style="color:Red">**0.978**</span>, which leads us to fail to reject the null hypothesis. Our observed data is not statistically unusual enough to verify the alternative hypothesis. As a result of this permutation test, we conclude that our model is fair and not biased towards the length of an ad.

## Code


```python
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics

%matplotlib inline
%config InlineBackend.figure_format = 'retina'  # Higher resolution figures
```

### Cleaning and EDA


```python
# importing data
eighteen_data = pd.read_csv('2018.csv')
nineteen_data = pd.read_csv('2019.csv')

# concatenate both datasets 
joined_data = pd.concat([eighteen_data, nineteen_data], ignore_index=True)

# convert time string to Timestamp object
joined_data['StartDate'] = pd.to_datetime(joined_data['StartDate'])
joined_data['EndDate'] = pd.to_datetime(joined_data['EndDate'])

# replace MALE and FEMALE with SINGLE
gender_dict = {'FEMALE': 'SINGLE', 'MALE': 'SINGLE'}
joined_data['Gender'] = joined_data['Gender'].replace(gender_dict)

# make the 'CountryCode' cleaner, less categories
joined_data['Cleaned_Country'] = joined_data['CountryCode'].apply(lambda x: 'other' if x != 'united states' else x) 

# create a column that says whether or not a company spent above the average
joined_data['More_Average'] = joined_data['Spend'] < np.mean(joined_data['Spend'])
joined_data['More_Average'] = joined_data['More_Average'].astype(int)

# get average time the Ad is up for
average_length = (joined_data['EndDate'] - joined_data['StartDate']).mean()

# impute the null values in 'EndDate' with 'StartDate' + Average Time Ad is up
only_nulls = joined_data[joined_data['EndDate'].isnull()]
joined_data['EndDate'] = joined_data['EndDate'].fillna(only_nulls['StartDate'] + average_length)
joined_data['EndDate'] = joined_data['EndDate'].apply(lambda x: x.round('S'))

# create new column that holds the amount of time each Ad was up
# engineered feature 1
joined_data['LengthAd'] = (joined_data['EndDate'] - joined_data['StartDate'])

# clean 'OrganizationName', combines various OrgNames to 'Other'
joined_data['Org_Counts'] = joined_data['OrganizationName'].replace(joined_data['OrganizationName'].value_counts().to_dict())
joined_data.loc[joined_data.Org_Counts <= 30, 'OrganizationName'] = "Other"

#convert time to hours
def convert_to_hours(length):
        days = length.days
        hours, remainder = length.seconds // 3600, length.seconds % 3600
        minutes, seconds = remainder // 60, remainder % 60
        total_hrs = (days * 24) + hours + (minutes/60) + (seconds/3600)
        return total_hrs
# apply helper function
joined_data['LengthAd (Hours)'] = joined_data['LengthAd'].apply(convert_to_hours)
```


```python
#checks to see if there are any missing organization names or spending values
sum(joined_data['OrganizationName'] == np.nan) + sum(joined_data['Spend'] == np.nan)
```




    0



### <span style="color:Red">**Plot #1**</span>


```python
#gathering number of ads bought by each company
counts = pd.DataFrame(joined_data['OrganizationName'].value_counts())
#plotting a histogram of number of ads bought
ax = counts.plot(kind='hist', title='Plot 1: Frequency of ad volume', legend=False, bins=10)
ax.set_xlabel('Number of Ads bought')
```




    Text(0.5, 0, 'Number of Ads bought')




![png](output_7_1.png)


### <span style="color:Red">**Plot #2**</span>


```python
#selects organization names and how much spent per ad for simplicity
organization_money = joined_data[['OrganizationName', 'Spend']]
#found average money spent on each ad by each organization
mean_money_spent = organization_money.groupby('OrganizationName').mean()
#merged the average money spent with the number of ads spent found above
money_vs_numads = mean_money_spent.merge(counts, on=mean_money_spent.index)
#removed outliers greater than 15000 spent on average
money_vs_numads = money_vs_numads.loc[money_vs_numads['Spend'] <15000]
money_vs_numads = money_vs_numads.reset_index(drop=True)


numadsplot = money_vs_numads.plot(kind='scatter', x='Spend', y='OrganizationName', title='Plot 2: Number of ads versus money spent on ads')
numadsplot.set_xlabel('Average money spent on ad')
numadsplot.set_ylabel('Number of ads bought')
```




    Text(0, 0.5, 'Number of ads bought')




![png](output_9_1.png)


### <span style="color:Red">**Plot #3**</span>


```python
# fill all NaNs with 'BOTH'
joined_data['Gender'] = joined_data['Gender'].fillna('BOTH')

# get count of ADs for each gender
gender_series = joined_data.groupby('Gender').size()

# making gender bar graph
gender_lst = gender_series.index.tolist()
count_lst = gender_series.values.tolist()
gender_df = pd.DataFrame({'Gender': gender_lst, 'Count': count_lst})
gender_graph = gender_df.plot(kind = 'bar', x='Gender', y='Count', rot=0, color = ['g', 'b', 'r'], title='Plot 3: Number of ads per gender')
```


![png](output_11_0.png)


### <span style="color:Red">**Plot #4**</span>


```python
def clean_ages(age):
    if age[-1] == '-':
        new_age = age[:-1] + '+'
        return new_age
    elif age[-1] == '+':
        if age[-2] == '+':
            new_age = age[:-1]  
            return new_age
        else:
            return age
    else:
        return age
```


```python
# fill all NaNs with 'All Ages'
joined_data['AgeBracket'] = joined_data['AgeBracket'].fillna('All Ages')

# Clean 'AgeBracket' column
joined_data['AgeBracket'] = joined_data['AgeBracket'].apply(clean_ages)

# get count of ADs for each age bracket
age = joined_data['AgeBracket']
only_ageplus = joined_data[age.str.contains('+', regex = False)]
age_series = only_ageplus.groupby('AgeBracket').size().sort_values(ascending=True)

# making age bar graph
age_lst = age_series.index.tolist()[7:] # keep only those with 20+, make graph more readable
size_lst = age_series.values.tolist()[7:] # keep only those with 20+, make graph more readable
age_df = pd.DataFrame({'Ages': age_lst, 'Count': size_lst})
age_graph = age_df.plot(kind = 'barh', x='Ages', y='Count', color = ['g', 'b', 'r'], legend = False, title='Plot 4: Number of ads per age bracket')
```


![png](output_14_0.png)


### <span style="color:Red">**Plot #5**</span>


```python
# helper function to convert all currencies to USD
def helper(row):
    if row['Currency Code'] == 'AUD':
        return row['Spend'] * .65
    if row['Currency Code'] == 'CAD':
        return row['Spend'] * .71
    if row['Currency Code'] == 'EUR':
        return row['Spend'] * 1.08
    if row['Currency Code'] == 'GBP':
        return row['Spend'] * 1.23
    return row['Spend']
```


```python
# apply helper on 'Spend' column
joined_data['Spend'] = joined_data.apply(helper, axis = 1)
```


```python
# # making spend/gender bar graph
spend_gender = joined_data[['Spend', 'Gender']]
f_m = spend_gender.loc[(spend_gender['Gender'] == 'FEMALE') | (spend_gender['Gender'] == 'MALE')]
grouped_gender = f_m.groupby('Gender').sum().reset_index()
grouped_gender.plot(kind = 'pie', y = 'Spend', labels=['Female', 'Male'], colors=['g', 'y'], title='Plot 5: Total Spent per gender')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9cef062810>




![png](output_18_1.png)


### <span style="color:Red">**Plot #6**</span>


```python
# grouping by 'AgeBracket' and summing the 'Spend'
age_spend = joined_data[age.str.contains('+', regex = False)].groupby('AgeBracket').sum()['Spend'].reset_index()

# plotting 'AgeBracket' vs 'Spend'
age_spend.plot(kind = 'scatter', x = 'AgeBracket', y = 'Spend', rot= 90, title='Plot 6: Total Spent per AgeBracket')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9cebe18550>




![png](output_20_1.png)


### Assessment of Missingness

<span style="color:Red">**"Is the Candidate Ballot Information missing at random dependent on the amount of spend?"**</span>


```python
# obtaining the observed value
obs = joined_data.assign(is_null=joined_data['CandidateBallotInformation'].isnull()).groupby('is_null')['Spend'].mean().diff().abs().iloc[-1]

# permutation test
means = []
for i in range(10000):
    # shuffle 'Spend' column
    shuffled_col = (
        joined_data['Spend']
        .sample(replace=False, frac=1)
        .reset_index(drop=True)
    )
    # assign shuffled 'Spend' column and 'CandidateBallotInformation' nulls 
    shuffled = (
        joined_data
        .assign(**{
            'spend_shuffled': shuffled_col,
            'is_null': joined_data['CandidateBallotInformation'].isnull()
        })
    )
    
    # calculate difference of means
    mean = shuffled.groupby('is_null')['spend_shuffled'].mean().diff().abs().iloc[-1]
    means.append(mean)
    
# calculate p-value, compare to observed
means= pd.Series(means)
pval = np.mean(means > obs)
pval
```




    0.633



<span style="color:Red">**"Are the Regions (Included) missing at random dependent on the amount of spend?"**</span>


```python
# obtaining the observed value
obs = joined_data.assign(is_null=joined_data['Regions (Included)'].isnull()).groupby('is_null')['Spend'].mean().diff().abs().iloc[-1]

# permutation test
means = []
for i in range(10000):
    # shuffle 'Spend' column
    shuffled_col = (
        joined_data['Spend']
        .sample(replace=False, frac=1)
        .reset_index(drop=True)
    )
    # assign shuffled 'Spend' column and 'Regions (Included)' nulls 
    shuffled = (
        joined_data
        .assign(**{
            'spend_shuffled': shuffled_col,
            'is_null': joined_data['Regions (Included)'].isnull()
        })
    )
    
    # calculate difference of means
    mean = shuffled.groupby('is_null')['spend_shuffled'].mean().diff().abs().iloc[-1]
    means.append(mean)
    
# calculate p-value, compare to observed
means= pd.Series(means)
pval = np.mean(means > obs)
pval
```




    0.0293



### Hypothesis Test

<span style="color:Red">**(First Permutation Test)**</span>


```python
# renamed columns for simplicity
money_vs_numads = money_vs_numads.rename(columns = {'key_0' : 'Organization Name', 'Spend': 'Avg Spent', 'OrganizationName': 'Number of ads bought'})


# set x to be the average number of ads bought
x = money_vs_numads['Number of ads bought'].mean()

# using total variation distance as our test statistic
def tvd(num1, num2):
    return np.abs(num1-num2)/2

# getting the observed average amount of money spent on ads for companies that bought less than and greater than x ads
obs_above = money_vs_numads.loc[money_vs_numads['Number of ads bought'] < x]['Avg Spent'].mean()
obs_below = money_vs_numads.loc[money_vs_numads['Number of ads bought'] > x]['Avg Spent'].mean()
obs = tvd(obs_above, obs_below)

# do 1000 permutation tests
repetitions = 10000
differences = []
for i in range(repetitions):
    # shuffling the average money spent to assess the null hypothesis
    shuffled_spent = money_vs_numads['Avg Spent'].sample(replace = False, frac=1).reset_index(drop=True)
    shuffled = money_vs_numads.assign(**{'Shuffled Avg Spent': shuffled_spent})


    under_x = shuffled.loc[shuffled['Number of ads bought'] < x]
    above_x = shuffled.loc[shuffled['Number of ads bought'] > x]
    under_x_avg = under_x['Shuffled Avg Spent'].mean()
    above_x_avg = above_x['Shuffled Avg Spent'].mean()
    # getting test statistic using this permutation's averages
    differences.append(tvd(under_x_avg, above_x_avg))

# calculating p value and displaying charts
pval = np.mean(differences >= obs)
pd.Series(differences).plot(kind='hist', title='Plot 7: Distribution of 10000 permutation tests on the Average Spent on Ads')
plt.scatter(obs, 0, color='r', s=40);
print('pval is', pval)
```

    pval is 0.4601



![png](output_28_1.png)


<span style="color:Red">**(Second Permutation Test)**</span>


```python
# add column to check whether or not ad is for both genders or just one
joined_data['Both_Genders'] = joined_data['Gender'] == 'BOTH'
perm_df = joined_data[['Spend', 'Both_Genders']]

group_means = (
        perm_df
        .groupby('Both_Genders')
        .mean()
        .loc[:, 'Spend']
    )
observed_val = group_means.diff().iloc[-1]
    
gender_differences = []
for i in range(10000):
    
    # shuffle the Spend column
    shuffled_spend = (
        perm_df['Spend']
        .sample(replace=False, frac=1)
        .reset_index(drop=True)
    )
    
    # put them in a table
    shuffled = (
        perm_df
        .assign(**{'Shuffled Spend': shuffled_spend})
    )
    
    # compute the two group differences
    group_means = (
        shuffled
        .groupby('Both_Genders')
        .mean()
        .loc[:, 'Shuffled Spend']
    )
    gender_difference = group_means.diff().iloc[-1]
    
    # add it to the list of results
    gender_differences.append(gender_difference)
```


```python
pd.Series(gender_differences).plot(kind='hist', density=True, alpha=0.8, title='Plot 8: Distribution of 10000 permutation tests on the Average Spent on Ads with Gender')
plt.scatter(observed_val, 0, color='red', s=40);
p_val = np.count_nonzero(gender_differences >= observed_val) / 10000
print('pval is', p_val)
```

    pval is 0.0016



![png](output_31_1.png)


### Baseline Model


```python
def ord_org(full_df):
    #ordinally encodes organization names
    orgs = full_df['OrganizationName'].unique().tolist()
    org_encoding = {y:x for (x,y) in enumerate(orgs)}
    full_df['OrganizationOrd'] = full_df['OrganizationName'].replace(org_encoding)
    return full_df[['OrganizationOrd']]

def get_length_ad(full_df):
    # helper function to convert DateTime object to just hours
    def convert_to_hours(length):
        days = length.days
        hours, remainder = length.seconds // 3600, length.seconds % 3600
        minutes, seconds = remainder // 60, remainder % 60
        total_hrs = (days * 24) + hours + (minutes/60) + (seconds/3600)
        return total_hrs
    # apply helper function
    full_df['LengthAd (Hours)'] = full_df['LengthAd'].apply(convert_to_hours)
    
    return full_df[['LengthAd (Hours)']]
```


```python
# creating pipelines for one hot encoding and ordinal encoding
convert_cat = Pipeline(steps = [('one_hot', OneHotEncoder(handle_unknown='ignore'))])
convert_org = Pipeline(steps = [('ord_org', FunctionTransformer(ord_org))])
# creating a column transformer for the pipelines
col_trans_baseline = ColumnTransformer(transformers = [
    ('categorical', convert_cat, ['Gender', 'Cleaned_Country']), 
    ('org', convert_org, ['OrganizationName'])])

neigh_baseline = KNeighborsClassifier(n_neighbors = 10)
# creating the base pipeline using KNeighborsClassifier
baseline_pipe = Pipeline([
    ('all_pips', col_trans_baseline), 
    ('kclass', neigh_baseline)]
)
```

### Final Model


```python
# creating a second engineered feature
# dividing the impressions by length of ad in hours 
def divide(full_df):
    full_df['LengthAd / Impressions'] =  full_df['Impressions'] / full_df['LengthAd (Hours)'] 
    return full_df[['LengthAd / Impressions']]

# multiply Impressions by 1
def multiply_impressions(full_df):
    full_df['Impressions'] =  full_df['Impressions'] * 1
    return full_df[['Impressions']]

# creating a column transformer for the pipelines
col_trans_final = ColumnTransformer(transformers = [
    # gets the total run time of the ad from the start and end times
    ('length', FunctionTransformer(get_length_ad), ['LengthAd']),
    # multiply 'Impressions' by 1 
    ('impressions', FunctionTransformer(multiply_impressions), ['Impressions']),
    # divides the length of the ad by the impressions it got
    ('div', FunctionTransformer(divide), ['LengthAd (Hours)', 'Impressions'])
])

neigh_final = KNeighborsClassifier(n_neighbors = 10)
# creating the final pipeline using KNeighborsClassifier
final_pip = Pipeline([
    ('all_pips', col_trans_final), 
    ('kclass', neigh_final)]
)
```

### Accuracy Assement of Both Models


```python
# accuracy assement of both models, n = 1000

# train both models:
# splitting the data
X_train, X_test, y_train, y_test = train_test_split(joined_data, joined_data['More_Average'])
# training baseline and final models on the same datset
baseline_pipe.fit(X_train, y_train)
final_pip.fit(X_train, y_train)

acc_base = []
acc_final = []

for i in range(1000):
    # randomizing the testing dataset
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(joined_data, joined_data['More_Average'])
    # getting accuracy of baseline and final models and appending it to their lists
    output_base = baseline_pipe.predict(X_test_final)
    output_final = final_pip.predict(X_test_final)
    
    acc_base.append(np.mean(output_base == y_test_final.values))
    acc_final.append(np.mean(output_final == y_test_final.values))
# returning mean accuracy of both models
np.mean(acc_base), np.mean(acc_final)
```




    (0.7887422680412371, 0.9460093720712277)



### Fairness Evaluation


```python
# add new column that either says if ad is short or long
joined_data['Classify_Ad_Length'] = (joined_data['LengthAd (Hours)'] > np.median(joined_data['LengthAd (Hours)'])).replace({True: 'long', False: 'short'})
```


```python
# re-fit the models to account for the new column added
baseline_pipe = baseline_pipe.fit(joined_data.drop('More_Average', axis = 1), joined_data['More_Average'])
final_pip = final_pip.fit(joined_data.drop('More_Average', axis = 1), joined_data['More_Average'])
```


```python
# split enitre dataset into 2 smaller subsets (short ad vs long ad)
short_ad = joined_data[joined_data['Classify_Ad_Length'] == 'short']
long_ad = joined_data[joined_data['Classify_Ad_Length'] == 'long']
# using our 'short ad' subset we predict values using our final model
short_preds = final_pip.predict(short_ad.drop('More_Average', axis = 1))
# we get the precision score
short  = metrics.precision_score(short_ad['More_Average'], short_preds)

# using our 'long ad' subset we predict values using our final model
long_preds = final_pip.predict(long_ad.drop('More_Average', axis = 1))
# we get the precision score
long = metrics.precision_score(long_ad['More_Average'], long_preds)

# get our observed value
obs = short - long
short, long, obs
```




    (0.9792221630261055, 0.9693763919821826, 0.009845771043922924)




```python
# permutation test
metrs = []
for _ in range(500):
    # shuffle the 'Classify_Ad_Length' column
    s = (
        joined_data
        .assign(Classify_Ad_Length=joined_data.Classify_Ad_Length.sample(frac=1.0, replace=False).reset_index(drop=True))
    )
    
    # break the larger dataset into 2 smaller subsets
    long_ad = s[s['Classify_Ad_Length'] == 'long']
    short_ad = s[s['Classify_Ad_Length'] == 'short']
    
    # predict using the 'long ad' subset and get the precision score
    long_preds = final_pip.predict(long_ad.drop('More_Average', axis = 1))
    long = metrics.precision_score(long_ad['More_Average'], long_preds)

    # predict using the 'short ad' subset and get the precision score
    short_preds = final_pip.predict(short_ad.drop('More_Average', axis = 1))
    short  = metrics.precision_score(short_ad['More_Average'], short_preds)
    
    # append all precision score differences to list
    metrs.append(short - long)
```


```python
# p-val, over our significance level of .05
pd.Series(metrs <= obs).mean()
```




    0.978




```python
pd.Series(metrs).plot(kind='hist', title='Permutation Test for Spent across Short/Long Ads')
plt.scatter(obs, 0, c='r');
```


![png](output_45_0.png)



```python

```
