import pandas as pd


df = pd.read_csv("CalCOFI.csv", nrows=1)

# Sütun isimlerini yazdır
print(df.columns.tolist())

# CSV dosyasını sadece gerekli sütunlarla ve ilk 30.000 satırla oku
df = pd.read_csv("CalCOFI.csv", usecols=["Depthm", "Salnty", "T_degC"], nrows=30000)

# Yeni veriyi kaydet
df.to_csv("filtered_bottle.csv", index=False)

