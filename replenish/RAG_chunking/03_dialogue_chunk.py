
from typing import List, Dict

def chunk_dialogue(turns: List[Dict], max_turns=10, max_chars=900, overlap_turns=2):
    """
    turns: [{"speaker":"User","text":"..." , "ts_start":123, "ts_end":130}, ...]
    """
    chunks = []
    i = 0
    while i < len(turns):
        j = i
        char_count = 0
        speakers = set()
        while j < len(turns):
            t = turns[j]
            uttr_len = len(t["text"])
            # 若单条超长，允许在句级二次切分（此处略），但不跨 speaker
            if (j - i + 1) > max_turns or (char_count + uttr_len) > max_chars:
                break
            char_count += uttr_len
            speakers.add(t["speaker"])
            j += 1

        if j > i:
            window = turns[i:j]
        elif i < len(turns):
            window = [turns[i]]
        else:
            break
        text = "\n".join([f'{t["speaker"]}: {t["text"]}' for t in window])
        meta = {
            "speakers": list(speakers),
            "turns_range": (i, j - 1),
            "ts_start": window[0].get("ts_start"),
            "ts_end": window[-1].get("ts_end"),
        }
        chunks.append({"text": text, "meta": meta})

        # 按轮次重叠回退
        if j >= len(turns):
            break
        next_start = i + len(window) - overlap_turns
        i = max(next_start, i + 1)  # 确保至少前进1步
    return chunks

turns = [
    {"speaker": "User",      "text": "今天天气真不错。",                "ts_start": 0,   "ts_end": 2},
    {"speaker": "Assistant", "text": "是啊，阳光正好。",                "ts_start": 3,   "ts_end": 5},
    {"speaker": "User",      "text": "要不要出去走走？",               "ts_start": 6,   "ts_end": 8},
    {"speaker": "Assistant", "text": "好主意，我正想出门透透气。",       "ts_start": 9,   "ts_end": 12},
    {"speaker": "User",      "text": "公园离这儿远吗？",               "ts_start": 13,  "ts_end": 15},
    {"speaker": "Assistant", "text": "不远，步行十分钟就到。",           "ts_start": 16,  "ts_end": 19},
    {"speaker": "User",      "text": "那我们现在就出发吧？",            "ts_start": 20,  "ts_end": 22},
    {"speaker": "Assistant", "text": "稍等，我拿一下水杯。",             "ts_start": 23,  "ts_end": 25},
    {"speaker": "User",      "text": "你带钥匙了吗？",                 "ts_start": 26,  "ts_end": 28},
    {"speaker": "Assistant", "text": "带了，在包里呢。",                 "ts_start": 29,  "ts_end": 31},
    {"speaker": "User",      "text": "路上小心点，人挺多的。",          "ts_start": 32,  "ts_end": 35},
    {"speaker": "Assistant", "text": "嗯，我会注意的。",                 "ts_start": 36,  "ts_end": 38},
    {"speaker": "User",      "text": "你看那只小狗多可爱！",            "ts_start": 39,  "ts_end": 42},
    {"speaker": "Assistant", "text": "真的！毛茸茸的，想摸摸它。",        "ts_start": 43,  "ts_end": 46},
    {"speaker": "User",      "text": "它主人好像在拍照。",              "ts_start": 47,  "ts_end": 49},
    {"speaker": "Assistant", "text": "我们别打扰他们了。",               "ts_start": 50,  "ts_end": 52},
    {"speaker": "User",      "text": "前面有家新开的咖啡馆。",          "ts_start": 53,  "ts_end": 56},
    {"speaker": "Assistant", "text": "待会儿回去可以试试。",             "ts_start": 57,  "ts_end": 59},
    {"speaker": "User",      "text": "记得你说喜欢抹茶拿铁？",          "ts_start": 60,  "ts_end": 63},
    {"speaker": "Assistant", "text": "对呀，希望他们家做得好喝。",        "ts_start": 64,  "ts_end": 67}
]

if __name__ == '__main__':
    '''
    max_turns_per_chunk：6–12 轮起步；语速快信息密度高可取 8–10。
    max_chars_per_chunk：600–1000 字；若存在长段独白，优先句级再切，不跨说话人。
    overlap_turns：1–2 轮；保证上一问下一答的连续性。
    '''
    chunks = chunk_dialogue(turns, max_turns=4, max_chars=90, overlap_turns=2)
    for chunk in chunks:
        print(chunk['text'])
        print('=' * 100)


