def fix_path_format(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    fixed_lines = [line.strip().replace('\\', '/') + '\n' for line in lines]

    with open(output_file, 'w') as f:
        f.writelines(fixed_lines)

# 修正兩個檔案
fix_path_format('train.txt', 'train_fixed.txt')
fix_path_format('val.txt', 'val_fixed.txt')

print("Path format fixed. New files: train_fixed.txt, val_fixed.txt")