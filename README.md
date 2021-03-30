# main.py
## 目的関数の指定
15行をアンコメント，16～20行をコメントアウトすると目的関数がsphere_offsetになる
16行をアンコメント，他の行をコメントアウトすると目的関数がktablet_offsetになる
17行をアンコメント，他の行をコメントアウトすると目的関数がbohachevsky_offsetになる
18行をアンコメント，他の行をコメントアウトすると目的関数がackley_offsetになる
19行をアンコメント，他の行をコメントアウトすると目的関数がschaffer_offsetになる
20行をアンコメント，他の行をコメントアウトすると目的関数がrastrigin_offsetになる

## 解法の指定
32～40行をアンコメント，32～70行の他の行をコメントアウトすると提案法のgoodを使わずbest1点のみを保存する方法を指定できる
42～50行をアンコメント，32～70行の他の行をコメントアウトすると提案法を指定できる
52～58行をアンコメント，32～70行の他の行をコメントアウトするとPSOを指定できる
60～70行をアンコメント，32～70行の他の行をコメントアウトすると提案法で学習し，個体の変数値をcsvに書き出す

42～58行をコメントアウトすると提案法とPSOの両方を指定できる

## 実行
$ python3 main.py

## 結果
例えば目的関数をsphere_offsetとしていた場合結果は benchmark/function=sphere_offset(20dim),n_eval=50000,n_loop=10/ に生成される
解法の指定で60～70行をアンコメントしていた場合には上に加えて benchmark/distribution/ にcsvが生成される
benchmark/function=sphere_offset(20dim),n_eval=50000,n_loop=10/ に生成されるのはヘッダ情報付きcsv
n_eval              目的関数の呼出回数
max_n_eval          目的関数の最大呼出回数(1ファイル内で常に一定の値)
dist_r              best_imgとoptimumとの距離
dist_stddev         dist_rの標準偏差
train_loss          トレーニングの損失値
fitness_mean        今世代に生成した個体の評価値の平均
fitness_best        今世代に生成した個体の評価値の最大値
fitness_best_so_far これまでに生成した全ての個体の評価値の最大値
n_p                 Generatorの出力個体数

## 結果の図示
benchmark/fitness_best_so_far_value.bat に書かれているコマンドでグラフの表示ができる．
生成されたbenchmark/function=sphere_offset(20dim),n_eval=50000,n_loop=10/をフォルダごとドラッグアンドドロップしても同コマンドを呼び出せる．

# 修論の図表の生成
上のmain.pyを用いる．

## 図5.1，図5.2
main.py 15行をアンコメント，16～20行をコメントアウトし目的関数をsphere_offsetにする
main.py 83～93行をアンコメントし提案法が実行されるようにする．32行以降の他の行はコメントアウトする．
$ python3 main.py
生成されたbenchmark/distribution/sphere_offset_before.csv，benchmark/distribution/sphere_offset_after.csvの1列目からエクセルでヒストグラムを表示

## 表5.2 1行目
main.py 15行をアンコメント，16～20行をコメントアウトし目的関数をsphere_offsetにする
main.py 44～52行，77～83行をアンコメントし提案法とPSOが実行されるようにする．32行以降の他の行はコメントアウトする．
$ python3 main.py
生成されたbenchmark/function=sphere_offset(20dim),n_eval=50000,n_loop=10/をフォルダごとbenchmark/fitness_best_so_far_value.batにドラッグアンドドロップ
解法ごとに，得られた解の目的関数値の平均が表示される

## 図5.4(a)
main.py 15行をアンコメント，16～20行をコメントアウトし目的関数をsphere_offsetにする
main.py 44～52行，77～83行をアンコメントし提案法とPSOが実行されるようにする．32行以降の他の行はコメントアウトする．
$ python3 main.py
生成されたbenchmark/function=sphere_offset(20dim),n_eval=50000,n_loop=10/をフォルダごとbenchmark/fitness_best_so_far.batにドラッグアンドドロップ
解法ごとに，得られた解の目的関数値の遷移が表示される．修論の同図と違いy軸の値は-1倍されている．

## 図5.5(a)
main.py 15行をアンコメント，16～20行をコメントアウトし目的関数をsphere_offsetにする
main.py 44～52行，77～83行をアンコメントし提案法とPSOが実行されるようにする．32行以降の他の行はコメントアウトする．
$ python3 main.py
生成されたbenchmark/function=sphere_offset(20dim),n_eval=50000,n_loop=10/をフォルダごとbenchmark/r.batにドラッグアンドドロップ
