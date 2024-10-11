import pandas as pd
import numpy as np
"""
data = {
    'Gün': ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14'],
    'Hava Durumu': ['Güneşli', 'Güneşli', 'Kapalı', 'Yağmurlu', 'Yağmurlu', 'Yağmurlu', 'Kapalı', 'Güneşli', 'Güneşli', 'Yağmurlu', 'Güneşli', 'Kapalı', 'Kapalı', 'Yağmurlu'],
    'Sıcaklık': ['Sıcak', 'Sıcak', 'Sıcak', 'Ilıman', 'Soğuk', 'Soğuk', 'Soğuk', 'Ilıman', 'Soğuk', 'Ilıman', 'Ilıman', 'Ilıman', 'Sıcak', 'Ilıman'],
    'Nem': ['Yüksek', 'Yüksek', 'Yüksek', 'Yüksek', 'Normal', 'Normal', 'Normal', 'Yüksek', 'Normal', 'Normal', 'Normal', 'Yüksek', 'Normal', 'Yüksek'],
    'Yağış': ['Seyrek', 'Aşırı', 'Seyrek', 'Seyrek', 'Seyrek', 'Aşırı', 'Aşırı', 'Seyrek', 'Seyrek', 'Seyrek', 'Aşırı', 'Aşırı', 'Seyrek', 'Aşırı'],
    'Oyun': ['Yok', 'Yok', 'Var', 'Var', 'Var', 'Yok', 'Var', 'Var', 'Yok', 'Var', 'Var', 'Yok', 'Var', 'Yok']
}

df = pd.DataFrame(data)

csv_file_path = "C:\\Users\\S\\projects\\Okul Ödevleri\\Yapay Zeka\\hava_durumu.csv"
df.to_csv(csv_file_path, index=False) #indexlenmesin diye false

df
"""
data = pd.read_csv("hava_durumu.csv")
df = pd.DataFrame(data)

#b) Pandas kütüphanesi aracılığıyla tablodan “Sıcaklık” ve “Nem” değerlerini siliniz (20p). 

df.drop(['Sıcaklık', 'Nem'], axis=1) #axis 1 sütün demek. ama drop'u böyle kullanırsa geçici silme işlemi yapar. kalıcı silmek istiyorsak inplace true yapmaliyiz

#c) Pandas kütüphanesinin metodu olan DataFrame()  ile yukarıda verilen tabloyu oluşturunuz ve 
#tablo hakkında betimleyici istatiksel bilgiler veriniz (20p).

df.info()
print(df.columns)

print(df)

#d) (3,4) boyutunda bir dizi oluşturunuz. Oluşturduğunuz bu dizinin boyutunu (6,2) olacak şekilde 
#değiştiriniz (20p). 

dizi = np.random.randint(100,size=(3, 4))
dizi.reshape(6,2)

print(dizi)

#e) İki tane (3,3) boyutunda rastgele sayılardan meydana bir dizi oluşturunuz. Oluşturulan bu diziyi 
#hem yatay hem de dikey olacak şekilde istif (stack) ediniz (20p). 

dizi1 = np.random.randint(100,size=(3, 3))
dizi2 = np.random.randint(100,size=(3, 3))

print(np.hstack((dizi1, dizi2))) #horizontal 
print(np.vstack((dizi1, dizi2))) #vertical
