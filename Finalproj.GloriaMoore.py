# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:54:47 2020

@author: Gloria
"""

from __future__ import print_function, division

import matplotlib.pyplot as plt

import numpy as np

import random

import thinkstats2
import thinkplot
import scipy.stats
import pandas as pd
import csv
import seaborn as sns
import statsmodels.formula.api as smf

factorname =''
mean = 0
vari = 0
std = 0

with open('2019.happiness.csv') as csvfile:
    hr2019 = pd.read_csv('2019.happiness.csv', delimiter = ',')
    
#Variable's descriptions and Histograms
def generatehist(dataframe, factorname):
    hist = dataframe[[factorname]].plot(kind='hist', rwidth=0.8, color='indianred')
    plt.xlabel('bins_value')
    plt.title('Histogram')
    plt.show()

def vardescrp(dataframe, factorname):
    mean = dataframe[[factorname]].mean()
    vari = dataframe[[factorname]].var()
    std = dataframe[[factorname]].std()
    mode = dataframe[[factorname]].mode()
    tail = dataframe[[factorname]].tail()
    print('mean', mean)
    print('variance', vari)
    print('std', std)
    print('mode', mode)
    print('tail', tail)
    return mean, vari, std



variablesh = ['Generosity', 'Freedom', 'GDP', 'Family', 'Life.exp', 'Corruption', "HappinessScore"]
for j in variablesh:
    generatehist(hr2019, j)
    vardescrp(hr2019, j)
#CDF Anlysis
cdf = thinkstats2.Cdf(hr2019.HappinessScore, label = 'Happiness Score')
thinkplot.Cdf(cdf)
thinkplot.Show(title='CDF', xlabel='Happiness Score', ylabel='CDF', color='orchid')

cdf = thinkstats2.Cdf(hr2019.Generosity, label = 'Generosity')
thinkplot.Cdf(cdf)
thinkplot.Show(title='CDF', xlabel='Generosity', ylabel='CDF', color='orchid')


cdf = thinkstats2.Cdf(hr2019.Family, label = 'Family')
thinkplot.Cdf(cdf)
thinkplot.Show(title='CDF', xlabel='Family', ylabel='CDF', color='orchid')


cdf = thinkstats2.Cdf(hr2019.Corruption, label = 'Corruption')
thinkplot.Cdf(cdf)
thinkplot.Show(title='CDF', xlabel='Corruption', ylabel='CDF', color='orchid')

cdf = thinkstats2.Cdf(hr2019.GDP, label = 'GDP')
thinkplot.Cdf(cdf)
thinkplot.Show(title='CDF', xlabel='GPD', ylabel='CDF', color='orchid')

happinessEurope = hr2019[hr2019.Region == 'Europe']
happinessnoEurope = hr2019[hr2019.Region != 'Europe']

happinessEurope[['HappinessScore']].plot(kind='density', color='blue')
plt.title('Happiness Score In European Countries')
plt.show()

hsEurope = np.array(happinessEurope['HappinessScore'])
pmf_hsEurope, bins_hsEurope = np.histogram(hsEurope, bins=10, density=True)


happinessnoEurope[['HappinessScore']].plot(kind='density', color='red')
plt.title('Happiness Score in the Rest of the Continents NO Europe')
plt.show()


hsnoEurope = np.array(happinessnoEurope['HappinessScore'])
pmf_hsnoEurope, bins_hsnoEurope = np.histogram(hsnoEurope, bins=10, density=True)



p1 = sns.distplot(hsEurope, hist = False, kde = True, label = 'Happiness Score Europe')
p1 = sns.distplot(hsnoEurope, hist = False, kde = True, label = 'Happiness Score NO Europe')
plt.title('Density Happiness Score Eurpean Countries and Rest of The World')
plt.ylabel('Density')
plt.show()

#Analytical Distributions
def MakeNormalModel(data):
    cdf = thinkstats2.Cdf(data, label='weights')
    mean, var = thinkstats2.TrimmedMeanVar(data)
    std = np.sqrt(var)
    print('n, mean, std', len(data), mean, std)

    xmin = mean - 4 * std
    xmax = mean + 4 * std
    xs, ps = thinkstats2.RenderNormalCdf(mean, std, xmin, xmax)
    thinkplot.Plot(xs, ps, label='model', linewidth=4, color='0.8')
    thinkplot.Cdf(cdf)
    thinkplot.Show()


def lognormaldist(data):
    log_x = np.log10(data)
    MakeNormalModel(log_x)
    thinkplot.Config(title='Lognormal Happiness Score', xlabel='Happiness Score',
                 ylabel='CDF')
    thinkplot.Show()

def MakeNormalPlot(data):
    mean, var = thinkstats2.TrimmedMeanVar(data, p=0.01)
    std = np.sqrt(var)
    xs = [-5, 5]
    xs, ys = thinkstats2.FitLine(xs, mean, std)
    thinkplot.Plot(xs, ys, color='0.8', label='model')
    xs, ys = thinkstats2.NormalProbability(data)
    thinkplot.Plot(xs, ys, label='Happiness Score')
    thinkplot.Show()

MakeNormalModel(hr2019.HappinessScore)
thinkplot.Config(title='Normal Happiness Score', xlabel='Happiness Score',ylabel='CDF')
MakeNormalPlot(hr2019.HappinessScore)
thinkplot.Config(title='Normal Probability Happiness Score', xlabel='Happiness Score',ylabel='CDF')

#Scatter plots
plt.scatter(hr2019.GDP, hr2019.HappinessScore, color='orchid')
plt.title('ScatterPlot Happiness Score Vs. GDP')
plt.ylabel('Happiness Score')
plt.xlabel('DGP')
plt.show()
plt.scatter(hr2019.Family, hr2019.HappinessScore, color='coral')
plt.title('ScatterPlot Happiness Score Vs. Family')
plt.ylabel('Happiness Score')
plt.xlabel('Family')
plt.show()
plt.scatter(hr2019[['Life.exp']], hr2019.HappinessScore, color='skyblue')
plt.title('ScatterPlot Happiness Score Vs. Life.exp')
plt.ylabel('Happiness Score')
plt.xlabel('Life.exp')
plt.show()

#Correlation Analysis
hr2019clean = hr2019[['HappinessScore', 'Generosity', 'Freedom', 'GDP', 'Family', 'Life.exp', 'Corruption']]
p2 = sns.pairplot(hr2019clean)
plt.title('Scatterplot Matrix')
plt.show()

corr = hr2019clean.corr()
sns.heatmap(corr, annot=True, vmin = -1, vmax = 1, center = 0, cmap=sns.diverging_palette(20, 220, n=50), square=True)
plt.show()

#Spearman's correlation for High Pearson's correlation
Sp_Fam_corr = thinkstats2.SpearmanCorr(hr2019.Family, hr2019.HappinessScore)
print('Spearman Correlation Happiness Score-Family:', Sp_Fam_corr)
Sp_Lifeexp_corr = thinkstats2.SpearmanCorr(hr2019['Life.exp'], hr2019.HappinessScore)
print('Spearman Correlation Happiness Score-Life.exp:', Sp_Lifeexp_corr)
Sp_GDP_corr = thinkstats2.SpearmanCorr(hr2019.GDP, hr2019.HappinessScore)
print('Spearman Correlation Happiness Score-GDP:', Sp_GDP_corr)
Sp_Corruption_corr = thinkstats2.SpearmanCorr(hr2019.Freedom, hr2019.HappinessScore)
print('Spearman Correlation Happiness Score-Corruption:', Sp_Corruption_corr)

#Correlation Hypothesis test
class CorrelationPermute(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        xs, ys = data
        test_stat = abs(thinkstats2.Corr(xs, ys))
        return test_stat

    def RunModel(self):
        xs, ys = self.data
        xs = np.random.permutation(xs)
        return xs, ys
    

data = hr2019.Family, hr2019.HappinessScore
ht = CorrelationPermute(data)
pvalue = ht.PValue()
print('pvalue of Correlation test Family-Happiness Score', pvalue)

data = hr2019.GDP, hr2019.HappinessScore
ht = CorrelationPermute(data)
pvalue = ht.PValue()
print('pvalue of Correlation test GDP-Happiness Score', pvalue)

data = hr2019['Life.exp'], hr2019.HappinessScore
ht = CorrelationPermute(data)
pvalue = ht.PValue()
print('pvalue of Correlation Life.exp-Happiness Score', pvalue)

data = hr2019.Corruption, hr2019.HappinessScore
ht = CorrelationPermute(data)
pvalue = ht.PValue()
print('pvalue of Correlation test Corruption-Happiness Score', pvalue)

#Diff in means Hypothesis test. Europe and no Europe

class DiffMeansPermute(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = abs(group1.mean() - group2.mean())
        return test_stat

    def MakeModel(self):
        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        self.pool = np.hstack((group1, group2))

    def RunModel(self):
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data
    
data = happinessEurope.HappinessScore, happinessnoEurope.HappinessScore
ht = DiffMeansPermute(data)
pvalue = ht.PValue()
print('pvalue Difference in mean of Happiness Score in European Countries and rest of the Countries', pvalue)

#Regression
hr2019.rename({'Life.exp': 'Life'}, axis=1, inplace=True)

formula = 'HappinessScore ~ Family + Life + GDP + Corruption'
model = smf.ols(formula, data=hr2019)
results = model.fit()
print(results.summary())






