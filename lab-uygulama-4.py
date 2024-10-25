#Soru-1)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

veri = pd.read_csv("train.csv")

label_encoder = LabelEncoder().fit(veri.price_range)
labels = label_encoder.transform(veri.price_range)
classes = list(label_encoder.classes_)

x = veri.drop(["price_range"], axis=1)
y = labels

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x = sc.fit_transform(x)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from tensorflow.keras.utils import to_categorical 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
model = Sequential()
model.add(Dense(16, input_dim=20, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(4, activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150)

import matplotlib.pyplot as plt

plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("Model Başarımları")
plt.ylabel("Başarım")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim", "Test"], loc="upper left")
plt.show()

plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.title("Model Kayıpları")
plt.ylabel("Kayıp")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim", "Test"], loc="upper left")
plt.show()


"""

#Soru-2-)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

veri = pd.read_csv("train.csv")

label_encoder = LabelEncoder().fit(veri.price_range)
labels = label_encoder.transform(veri.price_range)
classes = list(label_encoder.classes_)

x = veri.drop(["price_range"], axis=1)
y = labels

y = to_categorical(y, num_classes=4)

sc = StandardScaler()
x = sc.fit_transform(x)

def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim=x.shape[1], activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

model = KerasClassifier(model=build_model, epochs=150, batch_size=32, verbose=0)

kfold = KFold(n_splits=5, shuffle=True)
results = cross_val_score(model, x, y, cv=kfold)

print(f"Çapraz Doğrulama Ortalama Başarım: {results.mean() * 100:.2f}%")
print(f"Çapraz Doğrulama Standart Sapma: {results.std() * 100:.2f}%")

"""


#Soru-3)

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Veriyi yükleme
veri = pd.read_csv("train.csv")

# Çıkarılacak özellikler: "blue", "fc", "int_memory", "ram", "wifi"
features_to_drop = ["blue", "fc", "int_memory", "ram", "wifi"]
veri = veri.drop(features_to_drop, axis=1)

# Label Encoding
label_encoder = LabelEncoder().fit(veri.price_range)
labels = label_encoder.transform(veri.price_range)
classes = list(label_encoder.classes_)

# Özellik ve hedef değişkenler (features ve labels)
x = veri.drop(["price_range"], axis=1)  # Hedef değişken "price_range" çıkarılıyor
y = labels

# Hedef değişkenleri one-hot encoding
y = to_categorical(y, num_classes=4)

# Veriyi standartlaştırma
sc = StandardScaler()
x = sc.fit_transform(x)

# Çapraz doğrulama için gerekli kütüphaneler
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Modeli bir fonksiyon haline getirme
def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim=x.shape[1], activation='relu'))  # input_dim ayarı, yeni sütun sayısına göre yapıldı
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='softmax'))  # 4 sınıf için softmax aktivasyonu
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# Modeli KerasClassifier ile uyumlu hale getirme
model = KerasClassifier(model=build_model, epochs=150, batch_size=32, verbose=0)

# K-Fold Cross Validation (5 katlı çapraz doğrulama)
kfold = KFold(n_splits=5, shuffle=True)
results = cross_val_score(model, x, y, cv=kfold)

# Sonuçları değerlendirme
print(f"Çapraz Doğrulama Ortalama Başarım: {results.mean() * 100:.2f}%") # %31.15 çıktı.
print(f"Çapraz Doğrulama Standart Sapma: {results.std() * 100:.2f}%")import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

veri = pd.read_csv("train.csv")

features_to_drop = ["blue", "fc", "int_memory", "ram", "wifi"]
veri = veri.drop(features_to_drop, axis=1)

label_encoder = LabelEncoder().fit(veri.price_range)
labels = label_encoder.transform(veri.price_range)
classes = list(label_encoder.classes_)

x = veri.drop(["price_range"], axis=1)
y = labels

y = to_categorical(y, num_classes=4)

sc = StandardScaler()
x = sc.fit_transform(x)

def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim=x.shape[1], activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

model = KerasClassifier(model=build_model, epochs=150, batch_size=32, verbose=0)

kfold = KFold(n_splits=5, shuffle=True)
results = cross_val_score(model, x, y, cv=kfold)

print(f"Çapraz Doğrulama Ortalama Başarım: {results.mean() * 100:.2f}%")
print(f"Çapraz Doğrulama Standart Sapma: {results.std() * 100:.2f}%")


"""

#Soru-4)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Veri setini yükleme
veri = pd.read_csv("Diyabet-Veri Seti.csv")

# Özellikleri ve hedef değişkeni ayırma
x = veri.drop("class", axis=1)  # Özellikler
y = veri["class"]  # Hedef değişken

# Veriyi eğitim ve test setlerine ayırma (%80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# YSA Modeli oluşturma
model = Sequential()
model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))  # Girdi katmanı ve 12 nöronlu gizli katman
model.add(Dense(8, activation='relu'))  # Gizli katman
model.add(Dense(1, activation='sigmoid'))  # Çıktı katmanı (1 nöron, sigmoid aktivasyonu)

# Modeli derleme
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=10, verbose=0)

# Başarım ve kayıp grafikleri
plt.figure(figsize=(12, 5))

# Başarım grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Başarımları')
plt.ylabel('Başarım')
plt.xlabel('Epoch')
plt.legend(['Eğitim', 'Doğrulama'], loc='upper left')

# Kayıp grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Kayıpları')
plt.ylabel('Kayıp')
plt.xlabel('Epoch')
plt.legend(['Eğitim', 'Doğrulama'], loc='upper left')

plt.tight_layout()
plt.show()import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

veri = pd.read_csv("Diyabet-Veri Seti.csv")

x = veri.drop("class", axis=1)
y = veri["class"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=10, verbose=0)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Başarımları')
plt.ylabel('Başarım')
plt.xlabel('Epoch')
plt.legend(['Eğitim', 'Doğrulama'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Kayıpları')
plt.ylabel('Kayıp')
plt.xlabel('Epoch')
plt.legend(['Eğitim', 'Doğrulama'], loc='upper left')

plt.tight_layout()
plt.show()

"""

from sklearn.metrics import roc_curve, auc

y_pred_proba = model.predict(X_test).ravel()

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC Eğrisi (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.title('ROC Eğrisi')
plt.xlabel('Yanlış Pozitif Oranı (FPR)')
plt.ylabel('Doğru Pozitif Oranı (TPR)')
plt.legend(loc='lower right')
plt.grid()
plt.show()

