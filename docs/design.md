# data structure

- our spatial time graph based on a three-layers architecture, the first layer is original layer with original json, 第二层是语义层，语义层的每个节点指向原始层每个对它有贡献的节点，语义层是核心层

## original node
original json

## semantic node


---
- entity node:
    0. id: global unique
    1. entity type:{dynamic, static, attached}
    <!-- Let llm determine whether the object is static or dynamic based on its semantics. For the attached parts, if its layer_id > 1, it's attached. -->
    2. label:[(label, frame idx)]<!--Generally speaking, it's hard to change.-->
    3. attributes: [(attributes, frame idx)] <!--Generally speaking, it's hard to change.-->
    4. candidate: {"label": [（内容，帧号_idx）],"attribute": [], "type":[]} <!-- 这是在llm进行节点判断的时候，如果存在llm怀疑当前的type，label，attribute有不准的情况出现的时候，就将其认为可能的修改意见和对应的帧号_idx放进来；当然也可以删除（意味着根据后面的新信息，前面的怀疑是不对的）；也可以依据这个候选列表对于label，attribute，candidate进行修改，不过修改的时候一定要重新阅读引起怀疑的原始的json文档，确认应该修改,修改后将新的内容追加在上面对应字段的列表中-->
    5. last_matched: 
        
- static node(entity node):
    - first frame:
    - position : the coordinates of the detection box(If it occurs multiple times, take the average.)

    

- dynamic node(entity node):
    - first frame:
    - missed frame:
    - disappeared frame:
    - state : {active, inactive, disappeared}
    - trajectory : [(frame_idx, box, geometric center)]
    
    - life value:
    
    

- attached node(entity node):
    - owner: (id, label)


---

## semantic edge

- relation edge:
    1. id: global unique
    2. from_node_id:
    3. to_node_id:
    4. discribe:
    5. type:{static-static, static - dynamic, dynamic - dymanic, attached}
    6. valid_at:
    7. invalid_at: 


    # method
    get_id()
        全局范围内自增1

    insert(from_node_id, to_node_id, relation，frame id)
        insert edge into graph
        valid_at = frame id

    set




# function

```
# search candidate objecets based on spatial feature only
def get_candidates(idx): 

    # (1) filter 本帧疑似重复的cmp_objects

    """
    filter objects may be detected repeatedly in different label, for example:
    "idx": 4,
        "box": [
          0.198,
          39.444,
          205.297,
          140.599
        ],
        "score": 0.703,
        "label": "sign",
        "tag": "sign2",
        "attributes": "Sign with white letters on blue background",
     "idx": 59,
        "box": [
          0.212,
          39.865,
          205.107,
          139.416
        ],
        "score": 0.344,
        "label": "fabric",
        "tag": "fabric2",
        "attributes": "",
    "idx": 70,
        "box": [
            0.079,
            39.753,
            205.206,
            139.178
        ],
        "score": 0.686,
        "label": "poster",
        "tag": "poster",
        "attributes": "",
    actually, they represent the same thing, and it is an advertisement in fact.
    """

    the algorithm as follow:

    if IoU > threshold_cmp: append it into cmp_objects, the threshold should be very high, because only the objects in cmp_objects will be dedup;

    if cmp_objects.size() == 1: we get the only cmp_object, so that we needn't to get cur_context;
    else : we provide the cur_context for helping llm makes dicision
        如果几何中心距离不超过阈值（一定是大物体阈值高，小物体阈值低）按照IoU大小降序排列，并有limit参数控制最多加多少个 append it into cur_context

    if is_first_frame():
        return {"cur_cmp":[(idx, IoU，长，宽)], "cur_context": [(idx,IoU,distance，长，宽)],"pre":null}


    # (2) choose all the relevant entity nodes in graph 
    依据IoU和距离阈值检索图中相关节点，提供给llm做决策：
    前置条件：长宽比例不应该有太大的变化（选择尺寸差不多的物体）&&几何中心距离不应超过阈值（一定是大物体阈值高，小物体阈值低）

    筛选：IoU是首要指标，不管idx对应当前帧较大的检测框还是较小的检测框，只要检测到存储的图中的节点和idx的box的IoU大于阈值，就一定将其加入候选；对于较小（和整个画面的尺寸相比）的目标，若几何中心距离小于某个阈值，也是可以接受的
    进而得到所有的对应候选的entity node，提供给llm做决策

    return  {"cur_cmp":[(idx, IoU)], "cur_context": any,"pre":[(node id, IoU, 几何中心距离，长宽)]}

    
# llm 匹配处理与节点处理
def node_resolve(cmp_objects, cur_context, relevant_entity_nodes, frame_id):
    (1) 构建上下文：将疑似重复的本次cmp_objects和本次cur_context，前面的图中相似的entity node这三个集合的：label，box，attribute，candidates，相连的relations都放入上下文
    (2) 处理本帧重复：如果本帧的cmp_objects里面有多个object，则根据上下文来确定真实情况应该是什么label, 确定一个idx是正确的（不要新创造label，就是在这些里面挑一个相对最准确的就好），合并成一个object，记作uncertain_idx，同时还要判断性质：{dynamic, static, attached}，同时还要返回被认为不准确或重复的idx，便于未来修改这些objects对应的relations和标注这些idx已经经过处理，无需重复处理
    (3) 处理匹配问题: 对于在（2）中确定好的uncertain_idx，使用llm与先前图上的relevant entity nodes进行匹配。若relevant entity nodes为空，则代表图的语义层没有候选的能够匹配上的entity node，uncertain_idx是一个新的entity node，分配node id并记录node id和idx的映射, 并set last_matched = frame_id。如果存在查找到的relevant entity nodes，则用llm根据上下文匹配哪个entity node是uncertain_idx的正确匹配，如果成功匹配，则记录node id和idx的映射以便后面的关系分析；反之则说明是一个新entity node操作同前。

    (4)完善信息：
        (1)如果是一个新节点，则无论什么类型{dynamic, static, attached}都需要记录：node id, type, label ,attribute;
            - 如果是attached，先不用记录它的owner，一会在关系的处理上确定owner；
            - 如果是dynamic，要记录first frame，missed frame和disappeared frame设置为null，state设置为active， trajectory先放当前帧的就行；
            - 如果是static，要记录first frame，position
        (2) 
            1. 如果匹配上了，要进行一个reflection，也即对于type，label，attribute进行反思，看看是不是有可能不对，只有在明显感觉有问题的地方（比如把车识别成人这种级别的）才将llm认为的正确的答案写在node的对应的candidates里面，并要标注出来做出这个判断依据的是哪一帧的idx是啥的objects；
            2. 也可以反思之后撤销candidates里面的内容，认为前面错误怀疑了；
            3. 也可以反思之后觉得确实有问题，进行修改，修改方式就是在字段的对应内容后面进行追加（当前认为正确的内容和帧号），并记录发生了修改方便后续的边处理；
            4. 很有可能出现当前帧和图中对应的节点的label语义相同但形式不同的情况，如（poster和logo），对于static和attached，一般就以图中的为准，这种无需修改candidates，只需要记录一下是谁发生了改变就可以，便于后面的边进行处理；如果要以新帧为准，跳转3.
            5. 设置last_matched = frame_id，并设置state为Active, 设置life_value ++；
            6. 如果新旧节点的type不一致，不用特殊处理，就按照前面的反思流程走即可

            ！！如果发生变化一定要进行记录 node id和idx都要记录！！


# objects in current frame
def node_process(objects,frame id):
    for idx, object in objects:
        if is_process(idx):
            continue  # some objects may be processed due to others
        cmp_objects, cur_context, relevant_entity_nodes = get_candidates(idx)
        
        # 利用llm处理节点匹配
        node_resolve(cmp_objects, cur_context, relevant_entity_nodes, frame id)
    
    for entity_node in Graph:
        if entity_node.entity_type = "dynamic":
            if entity_node.last_matched!= frame id:
                entity_node.life_value--;
                if(entity_node.life_value>0);
                    entity_node.state = inactivate;
                else
                    entity_node.state = disappeared;
                
    
    

def edge_process(objects，frame_id)
    for entity_node in Graph:
    1. 构建上下文：将要处理的边的两个节点的所有边都拿出来，周围的相关节点等其他的有用信息
        if entity_node.last_matched == frame_id:
            """
            


            2. 对于当前帧的新边和图中的旧边的两个端点是都相同的情况
                1. 用llm检测新的边（对应idx的object的relations）是否和原有的边的表意重复，如果重复则直接丢弃
                2. 表意是否矛盾，如果矛盾则设置旧边的invalid_at为frame id，并插入新边
                3. 如果既不重复也不冲突，则直接将该边插入图中
            对于另一个端点不是的情况
                直接插入
            """
        else 直接插入

        1. 在插边的时候，有可能出现节点的idx在处理节点这一步的时候因为相似被合并，则需要修改对应的关系表述，例如原本的关系是“banner behind man”，但是banner和logo合并了，并确定了logo这个名字，则这个关系应该适应成“logo behind man”，这个由llm处理
        2. 另一种情况是，在插边的时候，出现了节点的label发生变化（这些在点处理的时候都会进行记录），也需要llm依据发生的变化进行边描述的对应修改

        3. 现在的边描述是这样的：sign1 displaying score for floor，将来节点的label后面可能有数字，将来的边描述不需要这个数字

        4. 顺手把attached的owner处理了

        

```

# 注意

1. static和attached偏向于稳定，各种改变尽量较少发生，因为他们就是视频的背景，如有冲突，尽量以旧的为准
2. idx在新帧中唯一，node id，edge id在时空图中唯一，用好这个唯一性进行关联，便于边的处理，便于处理各种改变后的级联反应
3. 这个和语言记忆的主要差异在于：
    1. 较为明显的静态动态之分
    2. 帧识别的准确性有问题
    3. 前后帧有较为明显的空间关联
    4. attached可以认为是固定的


# 总体流程

这就是通过输入的场景图数据来构建一个时空图谱，达到对于视频的记忆的目的，是一个memory系统

对于每一帧，先进行节点处理，再进行边处理，同时也进行向量嵌入，使用neo4j