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


def keep_first_n_frames(input_file, n_frames=2, output_file=None):
    """
    只保留JSON文件中的前n帧
    
    Args:
        input_file: 输入JSON文件路径
        n_frames: 保留的帧数，默认为2
        output_file: 输出JSON文件路径，如果为None则覆盖原文件
    """
    if output_file is None:
        output_file = input_file
    
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 只保留前n帧
    data = data[:n_frames]
    
    # 保存修改后的JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 已成功保留前{n_frames}帧")
    print(f"✓ 输入文件: {input_file}")
    print(f"✓ 输出文件: {output_file}")

def filter_first_frame_by_idxs(input_file, allowed_idxs, output_file=None, frame_index=0):
    """
    只保留指定帧（默认第0帧）中 idx 属于 allowed_idxs 的对象，保存为新文件或覆盖原文件。

    Args:
        input_file: 输入 JSON 路径
        allowed_idxs: 可迭代的 idx 集合（list/tuple/set），元素为 int 或可转换为 int
        output_file: 输出路径，默认覆盖输入文件
        frame_index: 要过滤的帧索引，默认 0（第一帧）
    """
    if output_file is None:
        output_file = input_file

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list) or frame_index >= len(data):
        raise ValueError("输入文件应为帧列表，且 frame_index 在范围内")

    allowed_set = set(int(x) for x in allowed_idxs)
    frame = data[frame_index]
    if 'objects' in frame and isinstance(frame['objects'], list):
        filtered = [obj for obj in frame['objects'] if int(obj.get('idx', -1)) in allowed_set]
        frame['objects'] = filtered

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✓ 第{frame_index}帧已保留 idx 在 {sorted(allowed_set)} 的对象")
    print(f"✓ 输入文件: {input_file}")
    print(f"✓ 输出文件: {output_file}")


    


if __name__ == '__main__':
    # 定义文件路径
    less_move_file = './data/less_move copy 2.json'
    output = './data/test/1.json'
    
    # 执行只保留前2帧的操作
    filter_first_frame_by_idxs(output,[4,59,70,2,44,49,24,14],frame_index=0)

