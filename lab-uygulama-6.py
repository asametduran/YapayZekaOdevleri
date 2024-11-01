import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

servis_kalitesi = ctrl.Antecedent(np.arange(0, 11, 1), 'servis_kalitesi')
yemek_kalitesi = ctrl.Antecedent(np.arange(0, 11, 1), 'yemek_kalitesi')
bahsis = ctrl.Consequent(np.arange(0, 26, 1), 'bahsis')

servis_kalitesi['zayıf'] = fuzz.trimf(servis_kalitesi.universe, [0, 0, 5])
servis_kalitesi['kabul_edilebilir'] = fuzz.trimf(servis_kalitesi.universe, [0, 5, 10])
servis_kalitesi['harika'] = fuzz.trimf(servis_kalitesi.universe, [5, 10, 10])

yemek_kalitesi['kotu'] = fuzz.trimf(yemek_kalitesi.universe, [0, 0, 5])
yemek_kalitesi['idare_eder'] = fuzz.trimf(yemek_kalitesi.universe, [0, 5, 10])
yemek_kalitesi['lezzetli'] = fuzz.trimf(yemek_kalitesi.universe, [5, 10, 10])

bahsis['dusuk'] = fuzz.trimf(bahsis.universe, [0, 0, 13])
bahsis['orta'] = fuzz.trimf(bahsis.universe, [0, 13, 25])
bahsis['yuksek'] = fuzz.trimf(bahsis.universe, [13, 25, 25])

kural1 = ctrl.Rule(servis_kalitesi['harika'] | yemek_kalitesi['lezzetli'], bahsis['yuksek'])
kural2 = ctrl.Rule(servis_kalitesi['kabul_edilebilir'], bahsis['orta'])
kural3 = ctrl.Rule(servis_kalitesi['zayıf'] & yemek_kalitesi['kotu'], bahsis['dusuk'])

bahsis_kontrol = ctrl.ControlSystem([kural1, kural2, kural3])
bahsis_hesaplama = ctrl.ControlSystemSimulation(bahsis_kontrol)

bahsis_hesaplama.input['servis_kalitesi'] = 7  
bahsis_hesaplama.input['yemek_kalitesi'] = 8   

bahsis_hesaplama.compute()
print(f"Önerilen Bahşiş Oranı: %{bahsis_hesaplama.output['bahsis']:.2f}")

servis_kalitesi.view()
yemek_kalitesi.view()
bahsis.view(sim=bahsis_hesaplama)
