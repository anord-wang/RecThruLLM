import random
def split_file_randomly(file_path, ratio=(8, 1, 1)):
        # 确保比例和为1
        total = sum(ratio)
        ratio = [r / total for r in ratio]
        
        # 读取文件并存储所有行
        with open(file_path, 'r', encoding='UTF-8') as file:
            lines = file.readlines()

        # 打乱行的顺序
        random.shuffle(lines)

        # 计算分割点
        length = len(lines)
        # length = 100
        split1 = int(length * ratio[0])
        split2 = split1 + int(length * ratio[1])
        print(split1, split2)
        print(length)

        # 分割列表
        group1 = lines[:split1]
        group2 = lines[split1:split2]
        group3 = lines[split2:]

        # 写入到新文件
        with open('/home/local/ASURITE/xwang735/LLM4REC/LLM4RecAgent/dataset/beauty/score_prediction_train.txt', 'w', encoding='UTF-8') as file:
            file.writelines(group1)
        with open('/home/local/ASURITE/xwang735/LLM4REC/LLM4RecAgent/dataset/beauty/score_prediction_val.txt', 'w', encoding='UTF-8') as file:
            file.writelines(group2)
        with open('/home/local/ASURITE/xwang735/LLM4REC/LLM4RecAgent/dataset/beauty/score_prediction_test.txt', 'w', encoding='UTF-8') as file:
            file.writelines(group3)

    # 调用函数，这里替换为你的文件路径
split_file_randomly('/home/local/ASURITE/xwang735/LLM4REC/LLM4RecAgent/dataset/beauty/score_prediction.txt')