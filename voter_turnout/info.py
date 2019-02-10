toDrop = ['id', 'HRMONTH', 'HRYEAR4', 'HUTYPEA', 'HUTYPC', 'HRHHID2', 'GESTCEN', \
            ]

probablyDrop = ['PEERNHRO', 'PUERN2', 'QSTNUM', 'OCCURNUM']

jobCodes = ['PEIO1ICD', 'PEIO1OCD', 'PEIO2ICD', 'PEIO2OCD', 'PRIMIND1', 'PRIMIND2',\
            ]

toDrop += probablyDrop


weights = ['PWFMWGT', 'PWLGWGT', 'PWORWGT', 'PWSSWGT', 'PWVETWGT', 'PRCHLD', 'PWCMPWGT',\
           'HWHHWGT']


questionable = ['HRLONGLK', 'PRTFAGE', 'PRFAMNUM', 'PRCITFLG',\
                'PUBUSCK1', 'PUBUSCK2', 'PUBUSCK3', 'PUBUSCK4', 'PUHRCK1', 'PUHRCK2',\
                'PUHRCK3', 'PUHRCK4', 'PUHRCK5', 'PUHRCK6', 'PUHRCK7', 'PUHRCK12',\
                'PULAYCK1', 'PULAYCK2', 'PULAYCK3', 'PUDWCK1', 'PUDWCK2', 'PUDWCK3',\
                'PUDWCK4', 'PUDWCK5', 'PUJHCK1', 'PUJHCK2', 'PUJHCK4', 'PUJHCK5',
                'PUIOCK1', 'PUIOCK2', 'PUIOCK3']

lineNumbers = ['HURESPLI', 'HUBUSL1', 'HUBUSL2', 'HUBUSL3', 'HUBUSL4', 'PEPARENT',\
                 'PESPOUSE', 'PULINENO', 'PELNDAD', 'PELNMOM', 'PECOHAB']

cities = ['GTCBSAST', 'GTMETSTA', 'GTINDVPC']

# realized that most columns are one hot. Thus, we should just define
# what shouldn't be one hotted.
# oneHot = ['HUFINAL', 'HUSPNISH', 'HETENURE', 'HEHOUSUT', 'HETELHHD']

# I realize some blanks should be one hotted. The first one is gender.
oneHot = ['PESEX']

dontHot = ['PTWK']

needToDealWithBlank = ['PENATVTY']  
gradient = ['HUFAMINC', 'PEAGE', 'PEEDUCA', 'PRFAMTYP', 'PRINUSYR', 'PEMLR', \
            'PEHRUSL1', 'PEHRUSLT', 'PUHROFF2', 'PUHROT2', 'PEHRACT1', 'PEHRACT2',\
             'PEHRACTT', 'PREMPHRS', 'PRHRUSL', 'PRJOBSEA', 'PRPTHRS', 'PEERN']
difficultGradient = ['PRWKSTAT', 'PEERNPER', 'PRERNWA', 'PRCHLD', 'PRERNHLY']
# These variables are sums of two variables. Obviously they are gradient variables
# as well. Since I made this pretty late, some variables in gradient should be here 
# but aren't
sums = ['PEHRACTT']

# gradient += sum

# change -1 to zero are all gradients
changeMinusOneToZero = ['PELAYDUR', 'PELKDUR', 'PRUNEDUR', 'PEERNWKP', 'PRNMCHLD']
simpleGradient = ['HRNUMHOU', 'HUPRSCNT', 'PELAYDUR']

needToCombine = {}
needToCombine['jobs'] = ['PEMJOT', 'PEMJNUM']

#Information below is expressed in other columns maybe drop?
repeated = ['PEHRFTPT']

noOneHot = dontHot + gradient + needToDealWithBlank + needToDealWithBlank \
            + difficultGradient + sums + changeMinusOneToZero + simpleGradient\
            + needToCombine['jobs'] + repeated + lineNumbers + weights + ['target']


