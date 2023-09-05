# 必要なライブラリのインポート
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt

%matplotlib inline


# 訓練データ用CSVの読み込み
train_data = pd.read_csv("train_data.csv")
train_data.head()



# テストデータ用CSVの読み込み
test_data = pd.read_csv("test_data.csv")
test_data.head()



# 犬の写真をグレースケール化して表示
sample_img1 = Image.open("dc_photos/train/dog-001.jpg").convert("L")
plt.imshow(sample_img1, cmap="gray")



# 猫の写真をグレースケール化して表示
sample_img2 = Image.open("dc_photos/train/cat-001.jpg").convert("L")
plt.imshow(sample_img2, cmap="gray")



# グレースケール化した写真をndarrayに変換してサイズを確認
sample_img1_array = np.array(sample_img1)
sample_img1_array.shape



# 訓練用の航空写真の読み込み
# ndarrayのデータを保管する領域の確保
train_len = len(train_data)
# 左右、上下、180度回転させたものを用意するため、4倍の容量を確保する
X_train = np.empty((train_len * 5, 5625), dtype=np.uint8)                #X_trainは、train_lenの4倍の行数、それぞれの行に、画素数(600×800) の画像データが入る2次元配列とする。
y_train = np.empty(train_len * 5, dtype=np.uint8)                          #y_trainは、train_lenの4倍の行数、それぞれの行に、ゴルフ場がある:1・なし:0の1つの数字が入る1次元配列とする。

# 画像ひとつひとつについて繰り返し処理
for i in range(len(train_data)):

    # 基の画像をndarrayとして読み込んで訓練データに追加
    name = train_data.loc[i, "File name"]                                  #train_dataのi行目のFile nameを取得し、nameに格納する。
    train_img = Image.open(f"dc_photos/train/{name}.jpg").convert("L")     #nameに格納されているFile nameの画像データを、convert("L")で白黒に変換する。
    train_img = np.array(train_img)                                        #numpyのndarrayに変換する。
    train_img_f = train_img.flatten()                                      #train_imgは2次元配列になっているので、flatten()を施して、1次元配列に直す。
    X_train[i] = train_img_f                                               #作成したデータを計測データのi行目に格納する。
    y_train[i] = train_data.loc[i, "DC"]                                   #train_dataのi行目のGC(ゴルフ場あり:1、なし:0)を教師データのi行目に格納する。

    # 左右反転させたものを訓練データに追加
    train_img_lr = np.fliplr(train_img)                                    #fliplrを施して、左(left)右(right)反転する。
    train_img_lr_f = train_img_lr.flatten()
    X_train[i + train_len] = train_img_lr_f
    y_train[i + train_len] = train_data.loc[i, "DC"]

    # 上下反転させたものを訓練データに追加
    train_img_ud = np.flipud(train_img)                                  #flipudを施して、上(up)下(down)反転する。
    train_img_ud_f = train_img_ud.flatten()
    X_train[i + train_len * 2] = train_img_ud_f
    y_train[i + train_len * 2] = train_data.loc[i, "DC"]

    # 180度回転させたものを訓練データに追加
    train_img_180 = np.rot90(train_img, 2)                               #rot90を施して、180度(反時計回りに90度×"2"、"2"はrot90の2番目の引数)回転する。
    train_img_180_f = train_img_180.flatten()
    X_train[i + train_len * 3] = train_img_180_f
    y_train[i + train_len * 3] = train_data.loc[i, "DC"]
    
    
    
    
    # テスト用の航空写真の読み込み

# ndarrayのデータを保管する領域の確保
test_len = len(test_data)
X_test = np.empty((test_len, 5625), dtype=np.uint8)
y_test = np.empty(test_len, dtype=np.uint8)

# 画像ひとつひとつについて繰り返し処理
for i in range(test_len):

    # ndarrayとして読み込んでテストデータに追加
    name = test_data.loc[i, "File name"]
    test_img = Image.open(f"dc_photos/test/{name}.jpg").convert("L")
    test_img = np.array(test_img)
    test_img_f = test_img.flatten()
    X_test[i] = test_img_f
    y_test[i] = test_data.loc[i, "DC"]
    
    
    
    
    
 # 分類器の作成
classifier = SVC(kernel="linear")
classifier.fit(X_train, y_train)



# 分類の実施と結果表示
y_pred = classifier.predict(X_test)
y_pred



# 正解の表示
y_test


# 混同行列で正答数の確認
print(metrics.confusion_matrix(y_test, y_pred))


print(metrics.accuracy_score(y_test, y_pred))


print(metrics.classification_report(y_test, y_pred))
