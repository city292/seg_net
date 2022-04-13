# seg_net

# 训练数据
训练数据组织形式参考 flielist.txt

即：
输入图绝对路径 空格 标签图绝对路径

# 开始训练
参考 train.py中的参数设置

python3 train.py --ngpus 2



# 训好了模型，如何使用

参考predict.py

示例
python3 predict.py --input_dir 输入图片路径  --res_dir 输出路径  --ckpt 模型文件路径  --ext .tif


