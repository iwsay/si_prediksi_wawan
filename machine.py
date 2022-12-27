import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

def trainTest():
    # Variabel independen
    x = Cryotherapy.drop(["label"], axis = 1)
    x.head()

    # Variabel dependen
    y = Cryotherapy["label"]
    y.head()

    print("Set Head")
    print(y.head())

    # Import train_test_split function
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123)
    return x_train, x_test, y_train, y_test

# input data
Cryotherapy=pd.read_csv("datatreningjadi.csv", sep=";")
# Menampilkan data
print("Data Info")
Cryotherapy.info()
print()
print("isEmpty",Cryotherapy.empty)
print("sizedata",Cryotherapy.size)

xtr, xtest, ytr, ytest = trainTest()
# Mengaktifkan/memanggil/membuat fungsi klasifikasi Naive bayes
modelnb = GaussianNB()
# Memasukkan data training pada fungsi klasifikasi naive bayes
nbtrain = modelnb.fit(xtr, ytr)
print("Traiing result", nbtrain.class_count_)
print("\n\n")


# Menentukan hasil prediksi dari x_test
y_pred = nbtrain.predict(xtest)
resultPrediction = y_pred 
print("x test result")
print(resultPrediction)


# Menentukan probabilitas hasil prediksi
nbtrain.predict_proba(xtest)
print("prediction probabilation")
print(nbtrain.predict_proba(xtest))
print("\n\n")

# import confusion_matrix model
confusion_matrix(ytest, y_pred)
print("matrix")
print( confusion_matrix(ytest, y_pred))
print("\n\n")

# Merapikan hasil confusion matrix
y_actual1 = pd.Series([1, 0,1,0,1,0,1,0,1,0,0,1,1,0,1,1,0,0], name = "actual")
y_pred1 = pd.Series([1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1], name = "prediction")
df_confusion = pd.crosstab(y_actual1, y_pred1)
df_confusion
print("clean up matrix")
print(df_confusion)
print("\n\n")


# Menghitung nilai akurasi dari klasifikasi naive bayes 
print("accuration")
print(classification_report(ytest,y_pred))
print("\n\n")




