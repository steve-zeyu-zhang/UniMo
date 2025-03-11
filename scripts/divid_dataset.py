import os
import random
from os.path import join as pjoin

def get_all_npy_files(data_dir, file_type='.npy'):
    file_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(file_type):
                subfolder = os.path.relpath(root, data_dir)
                relative_path = pjoin(subfolder, file)
                file_list.append(relative_path.replace(file_type, ""))
    return file_list

def split_data(file_list, val_ratio=0.05, test_ratio=0.15):
    total_files = len(file_list)
    
    random.shuffle(file_list)

    val_size = int(total_files * val_ratio)
    test_size = int(total_files * test_ratio)
    train_size = total_files - val_size - test_size

    val_files = file_list[:val_size]
    test_files = file_list[val_size:val_size + test_size]
    train_files = file_list[val_size + test_size:]

    return train_files, test_files, val_files

def write_to_file(file_list, filename):
    with open(filename, 'w') as f:
        for file in file_list:
            f.write(f"{file}\n")

if __name__ == '__main__':
    dataset_root = '/root/autodl-tmp/pcmrl-vis/dataset/Coyote'
    data_dir = pjoin(dataset_root, 'motions')
    all_file = pjoin(dataset_root, 'all.txt')
    test_file = pjoin(dataset_root, 'test.txt')
    train_file = pjoin(dataset_root, 'train.txt')
    val_file = pjoin(dataset_root, 'val.txt')
    train_val_file = pjoin(dataset_root, 'train_val.txt')

    # 获取所有的 .npy 文件路径
    all_files = get_all_npy_files(data_dir, '.bvh')

    # 写入所有文件到 all.txt
    write_to_file(all_files, all_file)

    # 划分数据集
    train_files, test_files, val_files = split_data(all_files, val_ratio=0.05, test_ratio=0.15)

    # 写入各个数据集文件
    write_to_file(train_files, train_file)
    write_to_file(test_files, test_file)
    write_to_file(val_files, val_file)

    # train + val 写入 train_val.txt
    train_val_files = train_files + val_files
    write_to_file(train_val_files, train_val_file)

    print(f"Total files: {len(all_files)}")
    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")
    print(f"Val files: {len(val_files)}")
