import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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


def box_width(box) -> float:
    """计算检测框宽度，自动裁剪为非负值。"""
    return max(0.0, float(box[2]) - float(box[0]))


def box_height(box) -> float:
    """计算检测框高度，自动裁剪为非负值。"""
    return max(0.0, float(box[3]) - float(box[1]))


def area(box) -> float:
    """计算检测框面积。"""
    return box_width(box) * box_height(box)


def iou(box_a, box_b) -> float:
    """计算两个检测框的 IoU。

返回值范围 [0, 1]。当并集为 0 时返回 0。
"""
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union = area(box_a) + area(box_b) - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union



    
def get_diff_iou(input, idx1,idx2,frame1,frame2):
    """
    求不同帧的特定idx的iou
    """
    with open(input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list) or frame1 >= len(data) or frame2 >= len(data):
        raise ValueError("输入文件应为帧列表，且 frame1 和 frame2 在范围内")

    obj1 = next((obj for obj in data[frame1].get('objects', []) if int(obj.get('idx', -1)) == idx1), None)
    obj2 = next((obj for obj in data[frame2].get('objects', []) if int(obj.get('idx', -1)) == idx2), None)

    if obj1 is None or obj2 is None:
        raise ValueError("未找到指定idx的对象")

    print(f"label:{obj1['label']},{obj2['label']}")
    ans = iou(obj1.get('box', []), obj2.get('box'))  
    print(f"✓ 已成功计算IoU: {ans}")

def get_area_ratio(input,idx,frame,w,h):
    """
    求特定idx占总体的面积比例
    """
    with open(input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    obj = next((obj for obj in data[frame].get('objects', []) if int(obj.get('idx', -1)) == idx), None)

    print(f"label:{obj['label']},frame:{frame}")
    area_ratio = area(obj.get('box', [])) / (w * h) if w > 0 and h > 0 else 0.0
    print(f"✓ 已成功计算面积比例: {area_ratio}")


def compute_embedding_similarity(input_file, data:list[tuple[str, str]], model_name='all-MiniLM-L6-v2'):
    """
    使用 SentenceTransformer 计算两个不同帧中对象的相似度
    
    将对象的 label 和 attributes 结合作为文本，使用预训练的 SentenceTransformer 模型
    生成嵌入向量，然后计算余弦相似度。
    
    Args:
        input_file: 输入JSON文件路径
        idx1: 第一帧中的对象索引
        frame1: 第一个对象所在的帧号
        idx2: 第二帧中的对象索引
        frame2: 第二个对象所在的帧号
        model_name: SentenceTransformer 模型名称，默认为 'all-MiniLM-L6-v2'
    
    Returns:
        float: 相似度值，范围 [0, 1]
    """
    # 加载 SentenceTransformer 模型
    print(f"✓ 加载模型: {model_name}")
    model = SentenceTransformer(model_name)
        
    for text1, text2 in data:
        print(f"✓ 计算文本相似度:")
        print(f"  Text 1: '{text1}'")
        print(f"  Text 2: '{text2}'")
        
        
        # 生成嵌入
        embeddings = model.encode([text1, text2])
        
        # 计算余弦相似度
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        print(f"✓ 文本相似度: {similarity:.4f}")
   
    


if __name__ == '__main__':
    import pandas as pd

    # Login using e.g. `huggingface-cli login` to access this dataset
    df = pd.read_parquet("hf://datasets/ellisbrown/OpenEQA/v0/test-00000-of-00001.parquet")