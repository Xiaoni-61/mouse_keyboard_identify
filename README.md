# mouse_keyboard_identify
 基于核学习的终端用户识别算法的设计与实现（鼠标、键盘）

### keyboard
对击键原始数据进行处理（提取统计特征）
### MouseFirst
对鼠标原始数据进行事件层面的划分
### Mouse
对已经得到的鼠标事件数据进行特征的统计
### train_Scrapped
已经弃用的训练过程，具体为直接使用sklearn的SVC进行多分类
### train_MutiKernel
最终使用的训练过程，具体为使用MKLpy的库进行多核学习，文件内为不同训练处理轮次的代码以及对应的日志文件