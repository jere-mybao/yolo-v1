set -x

python3 yolo_v1/utils/data_retrieval.py
python3 yolo_v1/utils/data_preprocess.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt
cp 2007_test.txt test.txt
mkdir -p old_txt_files
mv 2007* 2012* old_txt_files/

mkdir -p data/images/
mkdir -p data/labels/
mv data/VOCdevkit/VOC2007/JPEGImages/*.jpg data/images/                                      
mv data/VOCdevkit/VOC2012/JPEGImages/*.jpg data/images/                                      
mv data/VOCdevkit/VOC2007/labels/*.txt data/labels/                                          
mv data/VOCdevkit/VOC2012/labels/*.txt data/labels/

rm -rf data/VOCdevkit/
python3 yolo_v1/utils/data_generate.py
mv test.txt old_txt_files/
mv train.txt old_txt_files/
mv test.csv data/
mv train.csv data/
rm -r old_txt_files/