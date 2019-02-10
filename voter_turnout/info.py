toDrop = ['id', 'HRMONTH', 'HRYEAR4', 'HUTYPEA', 'HUTYPC', 'HRHHID2', 'GESTCEN', \
            ]

#Information below is expressed in other columns maybe drop?
repeated = ['PEHRFTPT']

#toDrop += repeated

questionable = ['MONTH-IN-SAMPLE', 'HRLONGLK', 'PRTFAGE', 'PRFAMNUM', 'PRCITFLG'\
                'PUBUSCK1', 'PUBUSCK2', 'PUBUSCK3', 'PUBUSCK4', 'PUHRCK1', 'PUHRCK2',\
                'PUHRCK3', 'PUHRCK4', 'PUHRCK5', 'PUHRCK6', 'PUHRCK7', 'PUHRCK12',\
                'PULAYCK1', 'PULAYCK2', 'PULAYCK3', 'PUDWCK1', 'PUDWCK2', 'PUDWCK3',\
                'PUDWCK4', 'PUDWCK5', 'PUJHCK1', 'PUJHCK2', 'PUJHCK4', 'PUJHCK5']

lineNumbers = ['HURESPLI', 'HUBUSL1', 'HUBUSL2', 'HUBUSL3', 'HUBUSL4', 'PEPARENT',\
                 'PESPOUSE', 'PULINENO']

cities = ['GTCBSAST', 'GTMETSTA', 'GTINDVPC']

# realized that most columns are one hot. Thus, we should just define
# what shouldn't be one hotted.
# oneHot = ['HUFINAL', 'HUSPNISH', 'HETENURE', 'HEHOUSUT', 'HETELHHD']

# I realize some blanks should be one hotted. The first one is gender.
oneHot = ['PESEX']

needToDealWithBlank = ['PENATVTY']  
gradient = ['HUFAMINC', 'PEAGE', 'PEEDUCA', 'PRFAMTYP', 'PRINUSYR', 'PEMLR', \
            'PEHRUSL1', 'PEHRUSLT', 'PUHROFF2', 'PUHROT2', 'PEHRACT1', 'PEHRACT2',\
             'PEHRACTT', 'PREMPHRS', 'PRHRUSL', 'PRJOBSEA', 'PRPTHRS']
veryDifficultGradient = ['PRWKSTAT']
# These variables are sums of two variables. Obviously they are gradient variables
# as well. Since I made this pretty late, some variables in gradient should be here 
# but aren't
sums = ['PEHRACTT']

# gradient += sum

changeMinusOneToZero = ['PELAYDUR', 'PELKDUR', 'PRUNEDUR']
simpleGradient = ['HRNUMHOU', 'HUPRSCNT', 'PELAYDUR']

needToCombine = {}
needToCombine['jobs'] = ['PEMJOT', 'PEMJNUM']


