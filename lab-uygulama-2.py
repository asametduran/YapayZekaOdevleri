"""
1-) Kullanıcıdan 3 adet integer türünde değer alınız. Almış olduğunuz bu değerler bir üçgenin 
açılarını ifade edecektir. Bu açı değerlerine göre bu üçgenin dik, geniş ya da dar üçgen olup 
olmadığını belirleyen programı yazınız (20p). 

aci1 = int(input("Ucgenin aci degerini giriniz:"))
aci2 = int(input("Ucgenin aci degerini giriniz:"))
aci3 = int(input("Ucgenin aci degerini giriniz:"))

if(aci1 == 90 or aci2==90 or aci3==90):
    print("Bu bir dik ucgen")
elif(aci1>90 or aci2>90 or aci3>90):
    print("Bu bir genis ucgen")
else:
    print("Bu bir dar ucgen")


"""


"""
2) İçinde uzaylı olan bir oyun geliştirdiğinizi düşünün. uzaylı_rengi isminde bir değişken oluşturun 
ve bu değişken string türünde değerler alsın. Bu değişkene kırmızı, yeşil ya da sarı 
değerlerinden birini klavyeden veriniz. Eğer uzaylının rengi yeşilse “Tebrikler, yeşil uzaylıya ateş 
ettiğiniz için 5 puan kazandınız” şeklinde bir çıktı veriniz. Eğer rengi yeşil değilse "Tebrikler, yeşil 
olmayan uzaylıya ateş ettiğiniz için 10 puan kazandınız" şeklinde çıktı veriniz. Senaryoya ait 
programı yazınız (20p).




uzayli_rengi = input("Uzayli rengini giriniz:")

if(uzayli_rengi == "yesil"):
    print("Tebrikler, yesil uzaylıya ates ettiniz ve 5 puan kazandiniz")
else:
    print("Tebrikler, yesil olmayan uzaylıya ates ettiniz ve 10 puan kazandiniz")

"""
"""
3) Bir önceki sorudaki örneğe dayanarak if-elif-else yapılarını kullanarak aşağıdaki soruları 
cevaplayınız (20p). 
a) Eğer uzaylı rengi yeşil ise "Tebrikler, yeşil uzaylıya ateş ettiğiniz için 5 puan kazandınız", 
b) Eğer uzaylı rengi sarı ise "Tebrikler, sarı uzaylıya ateş ettiğiniz için 10 puan kazandınız", 
c) Eğer uzaylı rengi kırmız ise "Tebrikler, kırmızı uzaylıya ateş ettiğiniz için 15 puan kazandınız" 
şeklinde çıktı veren programı yazınız. 


uzayli_rengi = input("Uzayli rengini giriniz:")

if(uzayli_rengi == "yesil"):
    print("Tebrikler, yesil uzaylıya ates ettiniz ve 5 puan kazandiniz")
elif(uzayli_rengi == "sari"):
    print("Tebrikler, sari uzaylıya ates ettiniz ve 10 puan kazandiniz")
else:
    print("Tebrikler, kirmizi uzayliya ates ettigniz icin 15 puan kazandiniz.") 
"""

"""
yas = int(input("Yaşınızı girin: "))

if yas < 2:
     print("Bu kişi bebektir.")
elif 2 <= yas <= 4:
    print ("Bu kişi yeni yürümeye başlayan çocuktur.")
elif 4 < yas <= 13:
    print  ("Bu kişi çocuktur.")
elif 13 <= yas < 20:
    print  ("Bu kişi ergendir.")
elif 20 <= yas <= 65:
    print  ("Bu kişi yetişkindir.")
else:
    print  ("Bu kişi yaşlıdır.")
"""
"""
f_meyveler = ["elma", "mandalina", "portakal", "erik", "hurma"]
o_meyveler = ["elma", "armut", "karpuz", "kavun", "muz", 
              "portakal", "çilek", "vişne", "kiraz", "mandalina"]

m_durumu = {}

for m in o_meyveler:
    if m in f_meyveler:
        m_durumu[m] = "Favori meyveler arasında."
    else:
        m_durumu[m] = "Favori meyveler arasında değil."

for m, d in m_durumu.items():
    print(f"{m}: {d}")

"""
