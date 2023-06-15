# handLangageTimeSformer
"handLangageTimeSformer"はTimeSformerを用いて手話単語の判別をするものです

# Usage
main.pyの
label2id, id2label = getLabel()
loadModelFineTune(label2id, id2label)
prepareDatasetsTraining()
trainModel()
sample_test_video = evaluate_test()
これを実行すると事前学習のデータセットが作り出されevaluate_test()で評価されます。
downloadDaset()はTimeSformerのデータセットなので手話とは関係ないですが、バスケなどの分類をすることが可能なデータセットです。
パスは自分のパスに変更して下さい
モジュールは各自でお願いします。m1は動きませんでした(2023 06)
