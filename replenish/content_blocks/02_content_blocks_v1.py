from langchain.messages import HumanMessage
from langchain_core.messages import AIMessage

# 消息内容
#   您可以将消息内容理解为发送到模型的数据有效载荷。消息具有一个content弱类型属性，支持字符串和无类型对象列表（例如字典）。
#   这使得 LangChain 聊天模型能够直接支持提供者原生结构，例如多模态内容和其他数据。



# String content
human_message = HumanMessage("你好?")

# 下面给出一个使用多模态输入的示例
# 提供者原生格式（例如OpenAI）
human_message = HumanMessage(content=[
    {"type": "text", "text": "你好?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
])

# 标准内容块 的列表
human_message = HumanMessage(content_blocks=[
    {"type": "text", "text": "你好?"},
    {"type": "image", "url": "https://example.com/image.jpg"},
])
# content_blocks在初始化消息时 进行指定仍然会填充消息content，但为此提供了一个类型安全的接口。

# -------------------------------------------------------------------------------- #


# 标准内容块：standard content blocks
# 消息对象实现了一个content_blocks属性，该属性会将属性值延迟解析content为标准的、类型安全的表示形式。
#    例如，ChatAnthropic或ChatOpenAI生成的消息将包含相应提供商格式的
#      `<Message>`thinking或reasoning`<Message>` 块，但可以延迟解析为一致的ReasoningContentBlock表示形式：

# anthropic格式
message = AIMessage(
    content=[
        {"type": "thinking", "thinking": "abcd", "signature": "abc123"},
        {"type": "text", "text": "jeff"},
    ],
    response_metadata={"model_provider": "anthropic"}
)
print(message.content_blocks)
# [{'type': 'reasoning', 'reasoning': 'abcd', 'extras': {'signature': 'abc123'}},
#  {'type': 'text', 'text': 'jeff'}]

# openai格式
message = AIMessage(
    content=[
        {
            "type": "reasoning",
            "id": "rs_abc123",
            "summary": [
                {"type": "summary_text", "text": "summary 1"},
                {"type": "summary_text", "text": "summary 2"},
            ],
        },
        {"type": "text", "text": "jeff", "id": "msg_abc123"},
    ],
    response_metadata={"model_provider": "openai"}
)
print(message.content_blocks)
# [{'type': 'reasoning', 'id': 'rs_abc123', 'reasoning': 'summary 1'},
#  {'type': 'reasoning', 'id': 'rs_abc123', 'reasoning': 'summary 2'},
#  {'type': 'text', 'text': 'jeff', 'id': 'msg_abc123'}]


# -------------------------------------------------------------------------------- #


# 多模态
#   多模态是指处理不同形式数据的能力，例如文本、音频、图像和视频。LangChain 包含可供不同提供商使用的标准数据类型。
#   聊天模型可以接受多模态数据作为输入，并生成相应的输出。
#   【 额外的键可以包含在内容块的顶层，也可以嵌套在内容块中"extras": {"key": value} 】

# 图像输入
'''
# From URL
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这张图片的内容."},
        {"type": "image", "url": "https://example.com/path/to/image.jpg"},
    ]
}

# From base64 data
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这张图片的内容."},
        {
            "type": "image",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "image/jpeg",
        },
    ]
}

'''

# PDF文档输入
'''
# From URL
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这个文档的内容."},
        {"type": "file", "url": "https://example.com/path/to/document.pdf"},
    ]
}

# From base64 data
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这个文档的内容."},
        {
            "type": "file",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "application/pdf",
        },
    ]
}

'''

# 音频输入
'''
# From base64 data
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这个音频的内容."},
        {
            "type": "audio",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "audio/wav",
        },
    ]
}
'''

# 视频输入
'''
# From base64 data
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这个视频的内容."},
        {
            "type": "video",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "video/mp4",
        },
    ]
}
'''


