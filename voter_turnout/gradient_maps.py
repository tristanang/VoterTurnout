import numpy as np
from voter_turnout import info

master = {}

lst = ['HUFAMINC', 'PEAGE', 'PEEDUCA', 'PRINUSYR', 'PEHRUSL1', 'PEHRUSLT', 'PUHROFF2',\
       'PUHROT2', 'PEHRACT1', 'PEHRACT2', 'PRHRUSL', 'PEERN', 'PEERNPER', 'PRERNWA',
       'PEMJNUM']

lst += info.changeMinusOneToZero

mapOnly = ['PREMPHRS']

mapOnly += info.simpleGradient

notDone = []

for column in info.noOneHot:
    if column not in lst and column not in mapOnly and column != 'target':
        notDone.append(column)

maps = {}

for column in lst:
    master[column] = {}

for column in mapOnly:
    maps[column] = {}

# 'HUFAMINC' Incomes
income = {}

income[1] = np.average(np.arange(0,5000))
income[2] = np.average(np.arange(5000,7500))
income[3] = np.average(np.arange(7500,10000))
income[4] = np.average(np.arange(10000,12500))
income[5] = np.average(np.arange(12500,15000))
income[6] = np.average(np.arange(15000,20000))
income[7] = np.average(np.arange(20000,25000))
income[8] = np.average(np.arange(25000,30000))
income[9] = np.average(np.arange(30000,35000))
income[10] = np.average(np.arange(35000,40000))
income[11] = np.average(np.arange(40000,50000))
income[12] = np.average(np.arange(50000,60000))
income[13] = np.average(np.arange(60000,75000))
income[14] = np.average(np.arange(75000,100000))
income[15] = np.average(np.arange(100000,150000))
income[16] = 200000

master['HUFAMINC'] = income

# PEAGE Age
age = {}
age[85] = 90

master['PEAGE'] = age

# PEEDUCA Education: We can't do nothing about it

# PRINUSYR How long you've been a citizen. Don't know what to do

# PEHRUSL1 Hours worked per week. IDK what to do.

# PEHRUSLT Another hours worked.

# PUHROFF2 Hours took off

# PUHROT2 Overtime work

# PEHRACT1 How many hours did you actually work?

# PEHRACT2 How many hours did you work at your other job

# PREMPHRS why did you miss work
lazy = {}

for i in range(12,23):
    lazy[i] = 12

maps['PREMPHRS'] = lazy

# PRHRUSL hours worked weekly
weekly = {}

weekly[1] = np.average(np.arange(0,21))
weekly[2] = np.average(np.arange(21,35))
weekly[3] = np.average(np.arange(35,40))
weekly[4] = 40
weekly[5] = np.average(np.arange(41,50))
weekly[6] = 50
weekly[7] = -2
weekly[8] = -3

master['PRHRUSL'] = weekly

# PEERNPER periodicity
periodicity = {}
periodicity[7] = -2

master['PEERNPER'] = periodicity

# PEMJNUM if more than one job how many jobs

toNormalize = [] + notDone + mapOnly + lst

if __name__ == '__main__':
    
    assert len(toNormalize) == len(info.noOneHot) - 1

    print("Tests passed")


