master = {}

lst = ['HUFAMINC', 'PEAGE', 'PEEDUCA', 'PRINUSYR', 'PEHRUSL1', 'PEHRUSLT', 'PUHROFF2',\
        'PUHROT2', 'PEHRACT1', 'PEHRACT2']

mapOnly = ['PREMPHRS']
maps = {}

for column in lst:
    master[column] = {}

for column in mapOnly:
    maps[column] = {}

# 'HUFAMINC' Incomes
income = {}


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


