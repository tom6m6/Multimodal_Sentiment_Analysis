# 主要是用来消除一些由于训练集标签差距悬殊的问题
import random

def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            guid, tag = line.strip().split(',')
            data.append((guid, tag))
    return data

# 统计数据中各个标签的数量
def count_tags(data):
    tag_counts = {'negative': 0, 'positive': 0, 'neutral': 0}
    for _, tag in data:
        tag_counts[tag] += 1
    return tag_counts

# 从数据中随机选择 x 条 positive 条目
def select_positive(data, x):
    positive_data = [item for item in data if item[1] == 'positive']
    selected_positive = random.sample(positive_data, min(x, len(positive_data)))
    return selected_positive

def save_data(data, file_path):
    with open(file_path, 'w') as file:
        file.write('guid,tag\n')
        for item in data:
            file.write(f"{item[0]},{item[1]}\n")

def main(train_file, x, output_file):
    data = read_data(train_file)
    tag_counts = count_tags(data)
    print("Tag Counts:", tag_counts)

    # 随机选择 x 条 positive 条目
    selected_positive = select_positive(data, x)
    print(f"Selected {min(x, len(selected_positive))} positive items.")
    # 将选定的 positive 条目与原始数据合并
    new_data = [item for item in data if item[1] != 'positive'] + selected_positive
    

    save_data(new_data, output_file)
    print(f"Saved new data to {output_file}.")

if __name__ == "__main__":
    x=1188
    train_file = "data/train.txt"
    output_file = "data/train1.txt"
    main(train_file, x, output_file)
