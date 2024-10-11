"""
deger = int(input("Bir sayı girin: "))
for i in range(1, 11):
    sonuc = deger * i
    print(f"{deger} * {i} = {sonuc}")
"""

"""
sayi = int(input("Bir sayı girin: "))
basamak_sayisi = 0

while sayi != 0:
    sayi //= 10
    basamak_sayisi += 1

print(f"Girilen sayı {basamak_sayisi} basamaklıdır.")
"""
"""
sayisalDeğerler = [12, 15, 32, 42, 55, 75, 122, 132, 150, 180, 200]
for deger in sayisalDeğerler:
    if deger % 5 == 0 and deger <= 150:
        print(deger, end=', ')
"""
"""

i = 0
while i < len(sayisalDeğerler):
    if sayisalDeğerler[i] % 5 == 0 and sayisalDeğerler[i] <= 150:
        print(sayisalDeğerler[i], end=', ')
    i += 1


"""

"""
a = int(input("a değerini girin: "))
b = int(input("b değerini girin: "))
c = int(input("c değerini girin: "))
sayac = 0

for i in range(a, b + 1):
    if i % c == 0:
        sayac += 1

print(f"{a} ile {b} arasında {c}'ye bölünebilen sayı sayısı: {sayac}")
"""

"""
for i in range(1, 100):
    print(f"{i} - {100 - i}")


"""

"""

ip = input("Bir IP adresi girin (örneğin 192.168.255.252): ")
parcalar = ip.split('.')
son = int(parcalar[3]) + 1

for i in range(5):
    if son > 255:
        son = 0
        parcalar[2] = str(int(parcalar[2]) + 1)
    print(parcalar[0] + '.' + parcalar[1] + '.' + parcalar[2] + '.' + str(son))
    son = son + 1

"""

