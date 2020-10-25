# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.datasets import load_iris
import math
from sklearn import metrics


iris=load_iris()
petala_comprimento=[]
petala_larg=[]
  
for n in range(len(iris.data)):

    petala_comprimento.append(iris.data[n,2])
    petala_larg.append(iris.data[n,3])
    
petala_larg=np.array(petala_larg)
petala_comprimento=np.array(petala_comprimento)

    
# New Antecedent/Consequent objects hold universe variables and membership
# functions
comp_petala = ctrl.Antecedent(np.arange(0,8, 0.1), 'comp_petala')
largura_petala = ctrl.Antecedent(np.arange(0, 3, 0.1), 'largura_petala')
irisc = ctrl.Consequent(np.arange(0, 3,0.1), 'irisc')
# um = ctrl.Consequent(np.arange(0, 1,0.1), 'um')
# dois = ctrl.Consequent(np.arange(0, 1,0.1), 'dois')


# Auto-membership function population is possible with .automf(3, 5, or 7)


# Custom membership functions can be built interactively with a familiar,
# Pythonic API


comp_petala['CTB'] = fuzz.trapmf(np.arange(0,8, 0.1), [0, 0,1.7, 2])
comp_petala['CTM'] = fuzz.trapmf(np.arange(0,8, 0.1), [2.4,3,5,5.3])
comp_petala['CTG'] = fuzz.trapmf(np.arange(0,8, 0.1), [4.3,4.5,7.2,7.2])


largura_petala['LTB'] = fuzz.trapmf(np.arange(0, 3, 0.1), [0, 0,0.7,0.9])
largura_petala['LTM'] = fuzz.trapmf(np.arange(0, 3, 0.1), [0.9,1,1.6,1.8])
largura_petala['LTG'] = fuzz.trapmf(np.arange(0, 3, 0.1), [1.4,1.5,2.7,2.7])


irisc['setosa'] = fuzz.trapmf(np.arange(0, 3, 0.1), [0,0,0.8,0.9])
irisc['versicolor'] = fuzz.trapmf(np.arange(0, 3, 0.1), [1,1.1,2,2.2])
irisc['virginica'] = fuzz.trapmf(np.arange(0, 3, 0.1), [1.9,2.3,3,3])


comp_petala.view()
largura_petala.view()
irisc.view()

rule1 = ctrl.Rule(comp_petala['CTB'] | largura_petala ['LTB'], irisc['setosa'])
rule2 = ctrl.Rule(comp_petala['CTM'] & largura_petala['LTM'], irisc['versicolor'])
rule3 = ctrl.Rule(comp_petala['CTG'] & largura_petala['LTM'], irisc['virginica'])
rule4 = ctrl.Rule(comp_petala['CTG'] & largura_petala['LTG'], irisc['virginica'])


"""
To help understand what the membership looks like, use the ``view`` methods.
"""

# You can see how these look with .view()



tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3,rule4])
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

resultado=[]
for i in range(0,150) :
    
    
    tipping.input['comp_petala'] = petala_comprimento[i]
    tipping.input['largura_petala']=petala_larg[i]
    tipping.compute()
    resultado.append(math.trunc(tipping.output['irisc']))

comp_petala.view(sim=tipping)
x=tipping.output['irisc']
irisc.view(sim=tipping)


resultado=np.array(resultado)
resultado=np.reshape(resultado,-1)
print(metrics.classification_report(iris.target,resultado,target_names=iris.target_names))
print (pd.crosstab(iris.target,resultado, rownames=['Real'], colnames=['          Predito'], margins=True))