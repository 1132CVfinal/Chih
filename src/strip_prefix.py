def strip_prefix(input_file, output_file, prefix="train_dataset/"):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    for line in lines:
        line = line.strip().replace("\\", "/")
        if line.startswith(prefix):
            line = line[len(prefix):]
        fixed_lines.append(line + '\n')

    with open(output_file, 'w') as f:
        f.writelines(fixed_lines)

strip_prefix("train.txt", "train_fixed.txt")
strip_prefix("val.txt", "val_fixed.txt")
