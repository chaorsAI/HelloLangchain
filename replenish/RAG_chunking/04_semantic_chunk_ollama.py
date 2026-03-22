
from typing import List, Dict, Tuple
import numpy as np
import re
import ollama

# 安装
# pip install ollama


def split_sentences_zh(text: str) -> List[str]:
    # 简易中文分句，可替换为 HanLP/Stanza 更稳健的实现
    pattern = re.compile(r'([^。！？；]*[。！？；]+|[^。！？；]+$)')
    return [m.group(0).strip()  for m in pattern.finditer(text)  if m.group(0).strip()]


def rolling_mean(vecs: np.ndarray, i: int, w: int) -> np.ndarray:
    s = max(0, i - w)
    e = min(len(vecs), i + w + 1)
    return vecs[s:e].mean(axis=0)

'''
语义分块:
分块策略：
    对⽂本先做句级切分，计算句⼦或短段的向量表示；
    当相邻语义的相似度显著下降（发⽣“语义突变”）时设为切分点。

参数调优说明（仅参考）
    阈值的含义：语义变化敏感度控制器，越低越容易切、越高越保守。
设定方式：
    绝对阈值：例如使用余弦相似度，若 sim < 0.75 则切分（需按语料校准）。
    相对阈值：对全篇的相似度/新颖度分布估计均值μ与标准差σ，使用 μ ± λσ 作为阈值，更稳健。
初始的配置建议（仅限于中文技术/说明文档）：
    窗口大小 window_size：2–4 句
    最小/最大块长：min_chunk_chars=300–400，max_chunk_chars=1000–1200
    阈值策略：novelty > μ + 0.8σ 或相似度 < μ - 0.8σ（先粗调后微调）
    overlap：10% 左右或按“附加上一句”做轻量轮次重叠
'''
def semantic_chunk(
        text: str,
        model_name: str = "bge-m3",
        window_size: int = 2,
        min_chars: int = 3,
        max_chars: int = 110,
        lambda_std: float = 0.8,
        overlap_chars: int = 8,
) -> List[Dict]:
    sents = split_sentences_zh(text)
    print("sents:", sents)
    if not sents:
        return []

    # 使用ollama
    emb = ollama.embed(model_name, sents)['embeddings']
    emb = np.asarray(emb)
    print("emb.shape:", emb.shape)

    # 基于窗口均值的“新颖度”分数
    novelties = []
    for i in range(len(sents)):
        ref = rolling_mean(emb, i- 1, window_size) if i > 0 else emb[0]  # 得到窗口
        ref = ref / (np.linalg.norm(ref) + 1e-8)  # 归一化
        novelty = 1.0 - float(np.dot(emb[i], ref))
        novelties.append(novelty)
    novelties = np.array(novelties)
    # print("novelties:", novelties)  # 新颖度

    # 相对阈值：μ + λσ
    mu, sigma = float(novelties.mean()), float(novelties.std() + 1e-8)
    threshold = mu + lambda_std * sigma
    # print("threshold: ", threshold)  相对阈值

    chunks, buf, start_idx = [], "", 0

    def flush(end_idx: int):
        nonlocal buf, start_idx
        if buf.strip():
            chunks.append({
                "text": buf.strip(),
                "meta": {"start_sent": start_idx, "end_sent": end_idx - 1}
            })
        buf, start_idx = "", end_idx

    for i, s in enumerate(sents):
        # 若超长则先冲洗
        if len(buf) + len(s) > max_chars and len(buf) >= min_chars:
            flush(i)
            # 结构化重叠：附加上一个块的尾部
            if overlap_chars > 0 and len(s) < overlap_chars:
                buf = s
                continue

        buf += s

        # 达到最小长度后遇到突变则切分
        if len(buf) >= min_chars and novelties[i] > threshold:
            flush(i + 1)

    if buf:
        flush(len(sents))

    return chunks


text = '''清晨六点，城市已经苏醒。地铁站口涌出第一批通勤者，他们低头盯着手机屏幕，脚步匆忙而机械。高楼林立的街道上，玻璃幕墙反射着刺眼的阳光，空气中弥漫着汽车尾气与咖啡混合的味道。人们在狭窄的人行道上擦肩而过，却彼此视若无物——仿佛每个人都被装进了一个透明的茧里，既看得见世界，又无法真正触碰它。

这种快节奏的生活看似高效，实则暗藏代价。长期处于高压状态会削弱人的专注力，甚至引发焦虑与失眠。研究表明，连续数周暴露在噪音、拥挤和信息过载的环境中，大脑前额叶皮层的活跃度会显著下降，这直接影响决策能力和情绪调节。更令人担忧的是，许多人已将这种状态视为“正常”，甚至以“忙碌”为荣，误以为疲惫是成功的勋章。

然而，人类并非生来就适应钢筋水泥的丛林。我们的祖先曾在广袤的自然中狩猎、采集、休憩，感官与四季节律紧密相连。即便在现代社会，身体深处仍保留着对自然的渴望。这一点在心理学中被称为“亲生命性”（biophilia）——一种与生俱来的亲近自然的本能。

于是，越来越多的人开始逃离城市，哪怕只是短暂地。周末驱车两小时，走进一片未被过度开发的山林，成了都市人疗愈心灵的秘密仪式。脚踩松软的腐殖土，耳畔是溪流潺潺与鸟鸣交织的白噪音，鼻腔里充盈着松针与湿润泥土的清香——这些细微的感官刺激，竟能神奇地平复神经系统的躁动。有实验显示，仅在森林中漫步两小时，人体皮质醇（压力激素）水平平均下降15%以上。

更有趣的是，自然不仅修复情绪，还能激发创造力。当人远离电子设备与待办清单，大脑默认模式网络（Default Mode Network）会被激活，这是产生灵感、整合记忆与自我反思的关键区域。许多作家、艺术家都坦言，他们最重要的构思往往诞生于散步、观云或静坐湖边的时刻。自然不是逃避现实的避难所，而是重新连接内在智慧的桥梁。

当然，并非所有人都有条件频繁接触荒野。但城市本身也在悄然改变。近年来，“口袋公园”“垂直绿化”“社区农园”等微更新项目在各大城市兴起。哪怕是一面爬满常春藤的墙，或阳台上的几盆薄荷，也能带来微妙的心理慰藉。关键不在于自然的规模，而在于是否建立了真实的互动——亲手浇水、观察植物生长、感受微风拂过树叶的节奏。

值得警惕的是，我们正用技术模拟自然，却可能错失其本质。虚拟现实中的森林漫游、白噪音APP里的雨声，虽然能暂时缓解焦虑，但终究是感官的“代餐”。真正的疗愈来自全身心的沉浸：皮肤感受阳光的温度，双脚感知地面的起伏，眼睛追踪一只蝴蝶的飞行轨迹。这种多模态的体验无法被算法完全复刻。

归根结底，现代生活的困境不在于城市本身，而在于我们与自然节律的断裂。重建这种连接，并非要回归原始，而是学会在水泥缝隙中种下绿意，在日程表里留出“无所事事”的空白。或许，真正的自由不是拥有更多选择，而是有能力慢下来，听见一片叶子落地的声音。

当夜幕降临，城市再次亮起霓虹。但这一次，有人关掉屏幕，推开窗户，深深吸了一口带着凉意的晚风。他知道，明天依然要挤地铁、回邮件、赶截止日期，但此刻，他允许自己只是存在——像一棵树那样，安静而坚定。
'''

if __name__ == '__main__':
    chunks = semantic_chunk(text, window_size=2, min_chars=20, max_chars=1000)

    for chunk in chunks:
        print(chunk['text'])
        print('=' * 100)

