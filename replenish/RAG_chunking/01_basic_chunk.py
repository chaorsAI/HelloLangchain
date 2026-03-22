''''''

'''
  1. 基于固定长度分块
'''
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="",        # 纯按长度切
    chunk_size=60,      # 依据实验与模型上限调整
    chunk_overlap=9,    # 15% 重叠
)

text = '''1.1 基于固定长度分块。
- 分块策略：按预设字符数 chunk_size 直接切分，不考虑文本结构。
- 优点：实现最简单、速度快、对任意文本通用。
- 缺点：容易破坏语义边界；块过大容易引入较多噪声，过小则会导致上下文不足。
- 适用场景：结构性弱的纯文本，或数据预处理初期的基线方案。'''
chunks = splitter.split_text(text)
# print(chunks)



'''
  2. 基于句子的分块
'''
import re

def split_sentences_zh(text: str):
    # 在句末标点（。！？；）后面带可选引号的场景断句
    pattern = re.compile(r'([^。！？；]*[。！？；]+|[^。！？；]+$)')
    sentences = [m.group(0).strip() for m in pattern.finditer(text) if m.group(0).strip()]
    return sentences


def sentence_chunk(text: str, chunk_size=600, overlap=80):
    sents = split_sentences_zh(text)
    chunks, buf = [], ""
    for s in sents:
        if len(buf) + len(s) <= chunk_size:
            buf += s
        else:
            if buf:
                chunks.append(buf)
                # 简单重叠：从当前块尾部截取 overlap 字符与下一句拼接
            buf = (buf[-overlap:] if overlap > 0 and len(buf) > overlap else "") + s
    if buf:
        chunks.append(buf)
    return chunks

text = '''1.2 基于句子的分块
- 分块策略：先按句子切分，再将若干句子聚合成满足chunk_size的块；保证最基本的语义完整性。
- 优点：句子级完整性最好。对问句/答句映射友好。便于高质量引用。
- 缺点：中文分句需特别处理。仅句子级切分可能导致块过短，需后续聚合。
- 适用场景：法律法规、新闻、公告、FAQ 等以句子为主的文本。
- 中文分句注意事项：
  - 不要直接用 NLTK 英文 Punkt：无法识别中文标点，分句会失败或异常。
  - 可以直接使用以下内容进行分句：
    - 基于中文标点的正则：按“。！？；”等切分，保留引号与省略号等边界。
    - 使用支持中文的 NLP 库进行更精细的分句：
    - HanLP（推荐，工业级，支持繁多语言学特性）Stanza（清华/斯坦福合作，中文支持较好）spaCy + pkuseg 插件（或 zh-core-web-sm/med/lg 生态）
    '''


chunks = sentence_chunk(text, chunk_size=60, overlap=9)
# print(chunks)


'''
    3. 基于递归字符分块
'''

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

separators = [
    r"\n#{1,6}\s",                 # 标题
    r"\n\d+(?:\.\d+)*\s",          # 数字编号标题 1. / 2.3. 等
    "\n\n",                        # 段落
    "\n",                          # 行
    " ",                           # 空格
    "",                            # 兜底字符级
]
splitter = RecursiveCharacterTextSplitter(
    separators=separators,
    chunk_size=70,
    chunk_overlap=10,
    is_separator_regex=True,       # 告诉分割器上面包含正则
)

text = '''# 1.3 基于递归字符分块
## 分块策略：给定一组由“粗到细”的分隔符（如段落→换行→空格→字符），自上而下递归切分，在不超出 chunk_size 的前提下尽量保留自然语义边界。
## 优点：在“保持语义边界”和“控制块大小”之间取得稳健平衡，对大多数文本即插即用。
## 缺点：分隔符配置不当会导致块粒度失衡，极度格式化文本（表格/代码）效果一般。
## 适用场景：综合性语料、说明文档、报告、知识库条目。'''

chunks = splitter.split_text(text)
print(chunks)




