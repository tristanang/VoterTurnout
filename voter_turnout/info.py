negatives = [-1, -2, -3, -9]

toDrop = ['id', 'HRMONTH', 'HRYEAR4', 'HUTYPEA', 'HUTYPC', 'HRHHID2', 'GESTCEN']

probablyDrop = ['PEERNHRO', 'PUERN2', 'QSTNUM', 'OCCURNUM', 'PRERNHLY']

# Missing columns in test_2008
missingColumns = ['HUTYPB']

# These variables are sums of two variables. Obviously they are gradient variables
# as well. Since I made this pretty late, some variables in gradient should be here
# but aren't
sums = ['PEHRACTT']

toDrop += probablyDrop

toDrop += sums

weights = ['PWFMWGT', 'PWLGWGT', 'PWORWGT', 'PWSSWGT', 'PWVETWGT', 'PRCHLD', 'PWCMPWGT',\
           'HWHHWGT']

lineNumbers = ['HURESPLI', 'HUBUSL1', 'HUBUSL2', 'HUBUSL3', 'HUBUSL4', 'PEPARENT',\
                 'PESPOUSE', 'PULINENO', 'PELNDAD', 'PELNMOM', 'PECOHAB']

boolean = ['PTWK']

# needToDealWithBlank = ['PENATVTY'] #no longer an issue

# Gradients I have dealt with
allGradient = []

# multiGroups = ['PRPTHRS']
# tooComplicated = ['PRWKSTAT']

gradient = ['HUFAMINC', 'PEAGE', 'PEEDUCA', 'PRINUSYR', \
            'PEHRUSL1', 'PEHRUSLT', 'PUHROFF2', 'PUHROT2', 'PEHRACT1', 'PEHRACT2',\
             'PREMPHRS', 'PRHRUSL', 'PEERN']
difficultGradient = ['PEERNPER', 'PRERNWA']

# change -1 to zero are all gradients
changeMinusOneToZero = ['PELAYDUR', 'PELKDUR', 'PRUNEDUR', 'PEERNWKP', 'PRNMCHLD']
simpleGradient = ['HRNUMHOU', 'HUPRSCNT']

needToCombine = {}
needToCombine['jobs'] = ['PEMJNUM'] #'PEMJOT'

#Information below is expressed in other columns maybe drop?
repeated = ['PEHRFTPT']

noOneHot = boolean + gradient \
            + difficultGradient + changeMinusOneToZero + simpleGradient \
            + needToCombine['jobs'] + lineNumbers + weights + ['target']

# Not used right now
# jobCodes = ['PEIO1ICD', 'PEIO1OCD', 'PEIO2ICD', 'PEIO2OCD', 'PRIMIND1', 'PRIMIND2']
# cities = ['GTCBSAST', 'GTMETSTA', 'GTINDVPC']
"""
questionable = ['HRLONGLK', 'PRTFAGE', 'PRFAMNUM', 'PRCITFLG',\
                'PUBUSCK1', 'PUBUSCK2', 'PUBUSCK3', 'PUBUSCK4', 'PUHRCK1', 'PUHRCK2',\
                'PUHRCK3', 'PUHRCK4', 'PUHRCK5', 'PUHRCK6', 'PUHRCK7', 'PUHRCK12',\
                'PULAYCK1', 'PULAYCK2', 'PULAYCK3', 'PUDWCK1', 'PUDWCK2', 'PUDWCK3',\
                'PUDWCK4', 'PUDWCK5', 'PUJHCK1', 'PUJHCK2', 'PUJHCK4', 'PUJHCK5',
                'PUIOCK1', 'PUIOCK2', 'PUIOCK3']
"""




