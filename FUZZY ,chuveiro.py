
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
x = np.arange(100)


#limites temperatura,ambiente
temp_agua = ctrl.Antecedent(np.arange(0, 41, 1), 'temp_agua')
temp_amb = ctrl.Antecedent(np.arange(0, 41, 1), 'temp_amb')
#limite abertura valvula próximo  de 20 à 90 %
valv = ctrl.Consequent(np.arange(0, 100, 1), 'valv')





#criação regreas inferencia
temp_agua['TF_agua'] = fuzz.trimf(temp_agua.universe, [0, 0,20])
temp_agua['TM_agua'] = fuzz.gaussmf(temp_agua.universe, 20, 5)
temp_agua['TQ_agua'] = fuzz.trimf(temp_agua.universe, [20, 40,40])


temp_amb['TF_amb'] = fuzz.trimf(temp_amb.universe, [0, 0,20])
temp_amb['TM_amb'] = fuzz.gaussmf(temp_amb.universe, 20, 5)
temp_amb['TQ_amb'] = fuzz.trimf(temp_amb.universe, [20, 40,40])

#valv['PA']= fuzz.trapmf(x, [0, 10, 30, 40])
valv['PA'] = fuzz.gaussmf(valv.universe, 20, 10)
valv['MA'] = fuzz.trimf(valv.universe, [30, 50, 70])
valv['TA'] = fuzz.gaussmf(valv.universe, 90, 10)



"""
To help understand what the membership looks like, use the ``view`` methods.
"""



temp_agua.view()
temp_amb.view()
valv.view()

#criação regras
rule1 = ctrl.Rule(temp_agua['TF_agua'] & temp_amb ['TF_amb'],valv['PA'])
rule2 = ctrl.Rule(temp_agua['TF_agua'] & temp_amb['TM_amb'], valv['PA'])
rule3 = ctrl.Rule(temp_agua['TF_agua'] & temp_amb['TQ_amb'], valv['MA'])
rule4 = ctrl.Rule(temp_agua['TM_agua'] & temp_amb['TF_amb'], valv['PA'])
rule5 = ctrl.Rule(temp_agua['TM_agua'] & temp_amb['TM_amb'], valv['MA'])
rule6 = ctrl.Rule(temp_agua['TM_agua'] & temp_amb['TQ_amb'], valv['TA'])
rule7 = ctrl.Rule(temp_agua['TQ_agua'] & temp_amb['TF_amb'], valv['MA'])
rule8 = ctrl.Rule(temp_agua['TQ_agua'] & temp_amb['TM_amb'], valv['TA'])
rule9 = ctrl.Rule(temp_agua['TQ_agua'] & temp_amb['TQ_amb'], valv['TA'])


tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3,rule4, rule5, rule6, rule7,rule8,rule9])
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)
tipping.input['temp_agua'] =10
tipping.input['temp_amb'] =20

tipping.compute()
temp_agua.view(sim=tipping)
temp_amb.view(sim=tipping)
valv.view(sim=tipping)

print('\n abetrura de : \n',tipping.output['valv'])
