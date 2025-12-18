set -x

python3 yolo_v1/utils/data_preprocess.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt
cp 2007_test.txt test.txt
mkdir -p old_txt_files
mv 2007* 2012* old_txt_files/
python3 yolo_v1/utils/data_generate.py
mv test.txt old_txt_files/
mv train.txt old_txt_files/
mv test.csv data/
mv train.csv data/
rm -r old_txt_files/