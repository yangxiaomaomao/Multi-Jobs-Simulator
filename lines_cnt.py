import os
import glob

def count_py_lines(directory):
    total_lines = 0

    # 遍历目录下的所有子目录和文件
    for filepath in glob.glob(os.path.join(directory, '**', '*.py'), recursive=True):
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            total_lines += len(lines)
        print(filepath, len(lines))
    return total_lines

# 设定你要统计的目录路径
directory_path = "."

# 统计并打印总行数
total_lines = count_py_lines(directory_path)
print(f"Total number of lines in .py files: {total_lines}")
