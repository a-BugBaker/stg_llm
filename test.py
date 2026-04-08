import json
import os

def remove_tag_field(input_file, output_file=None):
    """
    删除JSON文件中所有对象的tag字段
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径，如果为None则覆盖原文件
    """
    if output_file is None:
        output_file = input_file
    
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 删除tag字段
    for frame in data:
        if 'objects' in frame:
            for obj in frame['objects']:
                if 'tag' in obj:
                    del obj['tag']
    
    # 保存修改后的JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 已成功删除tag字段")
    print(f"✓ 输入文件: {input_file}")
    print(f"✓ 输出文件: {output_file}")


if __name__ == '__main__':
    # 定义文件路径
    less_move_file = './data/less_move.json'
    
    # 执行删除tag字段的操作
    remove_tag_field(less_move_file)
