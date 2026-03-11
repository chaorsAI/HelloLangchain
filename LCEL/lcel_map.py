"""
电商商品信息处理流水线 - RunnableMap
场景：处理用户上传的商品信息，生成标准化商品详情页
"""
import time
import json
from typing import Dict, Any, List
from datetime import datetime
from langchain_core.runnables import RunnableParallel, RunnableMap, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# 示例输入：用户上传的商品原始数据
raw_product_data = {
    "title": "Apple iPhone 15 Pro Max 256GB 原色钛金属 行货正品",
    "price": "8999.00",
    "seller": "Apple官方旗舰店",
    "specs": "6.7英寸, A17 Pro芯片, 4800万像素, 5倍长焦",
    "tags": "手机|苹果|iPhone|旗舰|5G",
    "description": "全新iPhone 15 Pro Max，采用航空级钛金属设计，史上最轻的Pro机型。",
    "timestamp": "2024-03-15T14:30:00Z"
}

print("=" * 80)
print("📦 原始商品数据:")
print(json.dumps(raw_product_data, indent=2, ensure_ascii=False))
print("=" * 80)

# ====================== 第一部分：RunnableMap 的核心作用 ======================

print("\n🔧 阶段1: RunnableMap - 数据清洗与标准化（顺序处理）")
print("-" * 60)


# 1.1 定义多个数据处理函数（这些都是顺序依赖的）
def extract_basic_info(data: Dict) -> Dict:
    """提取基础信息 - 假设我们需要品牌和型号"""
    title = data.get("title", "")
    # 简单解析品牌和型号
    if "iPhone" in title:
        brand = "Apple"
        model = "iPhone 15 Pro Max"
    elif "小米" in title or "Xiaomi" in title:
        brand = "Xiaomi"
        model = "未知型号"
    else:
        brand = "其他"
        model = "未知"

    return {
        "brand": brand,
        "model": model,
        "title_cleaned": title.replace("行货正品", "").strip()
    }


def parse_price_info(data: Dict) -> Dict:
    """解析价格信息 - 依赖基础信息中的品牌"""
    price_str = data.get("price", "0")
    try:
        price = float(price_str)
    except:
        price = 0.0

    # 根据品牌确定货币单位
    brand = data.get("brand", "未知")
    currency = "CNY" if brand in ["Apple", "Xiaomi", "华为"] else "USD"

    return {
        "price_numeric": price,
        "currency": currency,
        "price_formatted": f"{currency} {price:,.2f}"
    }


def categorize_product(data: Dict) -> Dict:
    """分类商品 - 依赖清理后的标题和价格"""
    title = data.get("title_cleaned", "")
    price = data.get("price_numeric", 0)

    category = "电子产品"
    subcategory = "手机"

    if price > 8000:
        price_segment = "高端"
    elif price > 3000:
        price_segment = "中端"
    else:
        price_segment = "入门"

    return {
        "category": category,
        "subcategory": subcategory,
        "price_segment": price_segment
    }

# 1.2 使用RunnableMap构建数据处理流水线
# 注意：这里每个步骤都依赖上一步的输出
data_processing_pipeline = RunnableMap({
    "basic_info": RunnableLambda(extract_basic_info),
    "price_info": RunnableLambda(lambda x: parse_price_info({**x, **extract_basic_info(x)})),
    "category_info": RunnableLambda(lambda x: categorize_product({
        **x,
        **extract_basic_info(x),
        **parse_price_info({**x, **extract_basic_info(x)})
    })),
    "original_data": RunnablePassthrough()
})

print("执行RunnableMap数据清洗...")
start_time = time.time()
processed_data = data_processing_pipeline.invoke(raw_product_data)
map_time = time.time() - start_time

print(f"\n✅ RunnableMap处理结果（耗时: {map_time:.3f}秒）:")
print("清洗后的数据:")
for key, value in processed_data.items():
    if key != "original_data":
        print(f"  {key}: {value}")

# ====================== 第二部分：RunnableParallel 的核心作用 ======================

print("\n\n🚀 阶段2: RunnableParallel - 并行丰富商品信息")
print("-" * 60)


# 2.1 定义多个可以并行执行的独立信息获取任务
def fetch_market_analysis(product_info: Dict) -> Dict:
    """模拟获取市场分析数据（慢速外部API）"""
    time.sleep(0.5)  # 模拟网络延迟
    brand = product_info.get("brand", "未知")
    model = product_info.get("model", "未知")

    return {
        "market_position": "行业领导者" if brand == "Apple" else "竞争者",
        "avg_market_price": 8799.00 if "iPhone 15" in model else 0.0,
        "price_trend": "稳定" if brand == "Apple" else "下降"
    }


def fetch_competitor_info(product_info: Dict) -> Dict:
    """模拟获取竞品信息（另一个慢速外部API）"""
    time.sleep(0.3)  # 模拟网络延迟
    category = product_info.get("category", "未知")

    competitors = {
        "手机": ["Samsung Galaxy S24", "Google Pixel 8", "小米14", "华为Mate 60"],
        "笔记本电脑": ["Dell XPS", "MacBook Pro", "ThinkPad"],
        "平板": ["iPad Pro", "Samsung Tab S9", "小米Pad 6"]
    }

    return {
        "main_competitors": competitors.get(category, ["暂无竞品信息"]),
        "competitor_count": len(competitors.get(category, []))
    }


def generate_seo_keywords(product_info: Dict) -> Dict:
    """生成SEO关键词（本地计算，较快）"""
    time.sleep(0.1)
    title = product_info.get("title_cleaned", "")
    brand = product_info.get("brand", "")
    model = product_info.get("model", "")

    # 提取关键词
    keywords = []
    if brand:
        keywords.append(brand)
    if model:
        keywords.append(model)

    # 添加通用关键词
    keywords.extend(["智能手机", "5G手机", "旗舰手机", "拍照手机"])

    return {
        "primary_keywords": keywords[:5],
        "long_tail_keywords": [f"{k}价格" for k in keywords[:3]] + [f"{k}评测" for k in keywords[:2]],
        "keyword_difficulty": 65  # SEO难度评分
    }


def check_inventory(product_info: Dict) -> Dict:
    """检查库存（数据库查询）"""
    time.sleep(0.2)
    model = product_info.get("model", "")

    inventory_status = {
        "iPhone 15 Pro Max": {"in_stock": True, "stock_count": 42, "delivery_days": 3},
        "小米14": {"in_stock": True, "stock_count": 156, "delivery_days": 1},
    }

    return inventory_status.get(model, {"in_stock": False, "stock_count": 0, "delivery_days": 14})


# 2.2 使用RunnableParallel并行执行所有信息获取任务
# 注意：所有任务接收相同的输入，独立执行
parallel_info_gathering = RunnableParallel({
    "market_data": RunnableLambda(fetch_market_analysis),
    "competitor_data": RunnableLambda(fetch_competitor_info),
    "seo_data": RunnableLambda(generate_seo_keywords),
    "inventory_data": RunnableLambda(check_inventory),
    "base_info": RunnablePassthrough()  # 传递基础信息
})

print("执行RunnableParallel并行信息收集...")
start_time = time.time()

# 组合处理后的数据 + 并行信息收集
full_processing_chain = (
        data_processing_pipeline  # 先顺序清洗数据
        | parallel_info_gathering  # 然后并行获取各种信息
)

parallel_result = full_processing_chain.invoke(raw_product_data)
parallel_time = time.time() - start_time

print(f"\n✅ RunnableParallel并行收集结果（总耗时: {parallel_time:.3f}秒）:")

# 计算并行任务各自的耗时
print("\n各并行任务模拟耗时:")
print(f"  - 市场分析: 0.5秒")
print(f"  - 竞品信息: 0.3秒")
print(f"  - SEO关键词: 0.1秒")
print(f"  - 库存检查: 0.2秒")
print(f"  → 如果串行执行总耗时: 0.5+0.3+0.1+0.2 = 1.1秒")
print(f"  → 实际并行执行耗时: {parallel_time - map_time:.3f}秒 (接近最慢任务的0.5秒)")

# ====================== 第三部分：两者结合的实际应用 ======================

print("\n\n🔗 阶段3: Map + Parallel 组合构建完整商品处理流水线")
print("-" * 60)


# 3.1 添加后续处理步骤（依赖并行收集的所有信息）
def generate_product_report(data: Dict) -> str:
    """生成完整的商品报告 - 依赖所有并行收集的信息"""
    basic = data.get("base_info", {}).get("basic_info", {})
    price = data.get("base_info", {}).get("price_info", {})
    category = data.get("base_info", {}).get("category_info", {})

    market = data.get("market_data", {})
    competitors = data.get("competitor_data", {})
    seo = data.get("seo_data", {})
    inventory = data.get("inventory_data", {})

    report = f"""
商品综合分析报告
{'=' * 50}

📋 基础信息:
- 品牌: {basic.get('brand', '未知')}
- 型号: {basic.get('model', '未知')}
- 价格: {price.get('price_formatted', 'N/A')}
- 分类: {category.get('category', '未知')} > {category.get('subcategory', '未知')}
- 价格段: {category.get('price_segment', '未知')}

📈 市场分析:
- 市场地位: {market.get('market_position', '未知')}
- 市场价格趋势: {market.get('price_trend', '未知')}
- 平均市场价格: {market.get('avg_market_price', 0):,.2f} {price.get('currency', 'CNY')}

🎯 竞争分析:
- 主要竞品: {', '.join(competitors.get('main_competitors', []))}
- 竞品数量: {competitors.get('competitor_count', 0)}个

🔍 SEO优化:
- 核心关键词: {', '.join(seo.get('primary_keywords', []))}
- 长尾关键词: {', '.join(seo.get('long_tail_keywords', []))}
- SEO难度指数: {seo.get('keyword_difficulty', 0)}/100

📦 库存与物流:
- 库存状态: {'有货' if inventory.get('in_stock') else '缺货'}
- 当前库存: {inventory.get('stock_count', 0)}件
- 预计发货: {inventory.get('delivery_days', 0)}个工作日
"""
    return report


# 3.2 完整的处理链：Map -> Parallel -> Map
complete_product_pipeline = (
        data_processing_pipeline  # Step 1: 数据清洗 (Map)
        | parallel_info_gathering  # Step 2: 并行收集 (Parallel)
        | RunnableLambda(generate_product_report)  # Step 3: 生成报告 (Map-like)
)

print("执行完整流水线(Map -> Parallel -> Map)...")
final_result = complete_product_pipeline.invoke(raw_product_data)
print(final_result)

# ====================== 第四部分：关键差异对比演示 ======================

print("\n" + "=" * 80)
print("📊 RunnableMap 与 RunnableParallel 关键差异对比")
print("=" * 80)

# 演示1: RunnableMap的数据依赖特性
print("\n1. 数据依赖演示")
print("   RunnableMap 示例（顺序依赖）:")


def step_a(data):
    print(f"    Step A: 接收 {data}, 输出 {data + 1}")
    return data + 1


def step_b(data):
    print(f"    Step B: 接收 {data}, 输出 {data * 2}")
    return data * 2


def step_c(data):
    print(f"    Step C: 接收 {data}, 输出 {data ** 2}")
    return data ** 2


# 错误的Map用法：试图并行但实际是顺序
print("   ❌ 错误理解（以为能并行）:")
try:
    wrong_map = RunnableMap({
        "a": RunnableLambda(step_a),
        "b": RunnableLambda(step_b),  # 这里需要data+1的结果，但实际接收的是原始输入5
        "c": RunnableLambda(step_c)  # 这里需要data*2的结果
    })
    wrong_result = wrong_map.invoke(5)
    print(f"     结果: {wrong_result}")  # 这不是我们想要的
except Exception as e:
    print(f"     错误: {e}")

print("\n   ✅ 正确的Map用法（显式传递依赖）:")
correct_map = RunnableLambda(lambda x: {
    "a_result": step_a(x),
    "b_result": step_b(step_a(x)),  # 显式调用step_a
    "c_result": step_c(step_b(step_a(x)))  # 显式调用前两步
})
print("     执行过程:")
result = correct_map.invoke(5)
print(f"     最终结果: {result}")

# 演示2: RunnableParallel的真正并行
print("\n2. 真正并行演示")
print("   创建3个独立任务，分别休眠不同时间:")


def task_fast(x):
    time.sleep(0.3)
    return f"fast({x})"


def task_medium(x):
    time.sleep(0.5)
    return f"medium({x})"


def task_slow(x):
    time.sleep(0.8)
    return f"slow({x})"


parallel_demo = RunnableParallel({
    "fast": RunnableLambda(task_fast),
    "medium": RunnableLambda(task_medium),
    "slow": RunnableLambda(task_slow)
})

print("   开始并行执行（理论上总耗时≈最慢的0.8秒）...")
start = time.time()
parallel_demo_result = parallel_demo.invoke(10)
end = time.time()
print(f"   实际耗时: {end - start:.2f}秒")
print(f"   结果: {parallel_demo_result}")

# 演示3: 错误使用场景对比
print("\n3. 常见错误使用场景")
print("   ❌ 错误: 用Parallel处理有依赖关系的数据")
print("     假设: 任务B需要任务A的结果，任务C需要任务B的结果")
print("     如果使用Parallel，B和C拿不到A的结果，会出错")

print("\n   ❌ 错误: 用Map执行独立的外部API调用")
print("     假设: 需要调用3个独立的外部API获取天气、新闻、股票")
print("     如果使用Map，会串行执行，总耗时 = 3个API耗时之和")
print("     应该用Parallel，总耗时 ≈ 最慢API的耗时")

# ====================== 第五部分：架构师最佳实践总结 ======================

print("\n" + "=" * 80)
print("🏆 架构师总结：如何正确选择 RunnableMap vs RunnableParallel")
print("=" * 80)

summary = """
🔑 核心决策树：

if 任务间有数据依赖关系:
    ├── 依赖是线性的（A→B→C）：使用多个RunnableLambda顺序组合
    ├── 依赖是复杂的：使用RunnableMap进行结构化转换
    └── 但记住：RunnableMap输出是单个字典，不是并行执行

elif 任务完全独立，可以同时执行:
    ├── 且需要合并所有结果：使用RunnableParallel
    ├── 外部API调用、数据库查询、文件读取：优先用Parallel
    └── 注意：所有分支接收相同输入

else:  # 混合场景
    ├── 先用Map清洗/准备数据
    ├── 然后用Parallel并发独立操作
    └── 最后用另一个Map整合结果

💡 关键洞察：

1. RunnableMap的本质是「数据转换器」
   - 输入: 任何类型 → 输出: 单个字典
   - 常用于: 数据清洗、特征提取、格式标准化
   - 不是并行执行！是按定义顺序执行

2. RunnableParallel的本质是「任务分发器」
   - 输入: 复制给所有分支 → 输出: 字典（每个分支一个键）
   - 真正并行执行，性能提升明显
   - 但分支间不能有数据依赖

3. 性能优化黄金法则：
   - IO密集型操作（网络请求、数据库查询）→ 优先考虑Parallel
   - CPU密集型操作（复杂计算）→ 考虑Python的GIL限制，可能用多进程
   - 数据预处理（清洗、转换）→ 用Map或顺序Lambda

4. 实际项目模式：
   【数据输入】
       ↓
   RunnableMap（清洗、验证、标准化）
       ↓
   RunnableParallel（并发获取外部数据）
       ↓
   RunnableMap（整合、丰富、格式化）
       ↓
   【结果输出】

⚠️ 常见陷阱：
1. 误以为RunnableMap能并行：它只是定义字典结构，不是并行执行
2. 在Parallel分支间传递数据：做不到！每个分支只看到原始输入
3. 过度使用Lambda：复杂逻辑应封装为独立函数，便于测试
4. 忽略错误处理：Parallel中一个分支失败可能影响整个并行块

✅ 最佳实践：
1. 为每个Runnable命名：.with_config(run_name="描述")
2. 添加超时和重试：特别是Parallel中的外部调用
3. 使用LangSmith追踪：清晰看到Map和Parallel的执行流
4. 编写单元测试：分别测试每个Runnable，再测试组合链
"""

print(summary)
print("=" * 80)