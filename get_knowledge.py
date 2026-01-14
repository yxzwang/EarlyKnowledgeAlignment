import os
import json
import argparse
import requests
from typing import List, Dict, Any
from tqdm import tqdm

class GraphR1Client:
    """Graph-R1 API客户端，能够处理双重编码的JSON响应并返回知识及其相关性分数。"""

    def __init__(self, base_url: str = "http://localhost:9001"):
        self.base_url = base_url
        self.search_endpoint = f"{base_url}/search"
        print(f"GraphR1Client initialized to connect to: {self.search_endpoint}")

    def _extract_all_knowledge(self, response_str: str) -> List[Dict[str, Any]]:
        """
        解析单个双重编码的JSON字符串，并提取所有知识及其相关性分数。
        
        Args:
            response_str: API返回的列表中的单个字符串元素。
            
        Returns:
            一个包含知识字典（含 'knowledge' 和 'coherence'）的列表，如果失败则返回空列表。
            例如: [{'knowledge': '...', 'coherence': 2.0}, {'knowledge': '...', 'coherence': 0.8}]
        """
        try:
            data = json.loads(response_str)
            results_list = data.get("results", [])
            
            if not results_list or not isinstance(results_list, list):
                return []
            
            # 提取知识和相关性分数
            knowledge_with_coherence = []
            for item in results_list:
                # 确保两个关键字段都存在
                if '<knowledge>' in item and '<coherence>' in item:
                    try:
                        # 尝试将coherence转换为浮点数以便排序
                        knowledge_with_coherence.append({
                            'knowledge': item.get('<knowledge>', ''),
                            'coherence': float(item.get('<coherence>'))
                        })
                    except (ValueError, TypeError):
                        # 如果coherence不是有效的数字，则赋予一个低优先级
                        knowledge_with_coherence.append({
                            'knowledge': item.get('<knowledge>', ''),
                            'coherence': -1.0 
                        })

            return knowledge_with_coherence

        except json.JSONDecodeError:
            print(f"警告: 无法解析内部JSON字符串: '{response_str[:100]}...'")
            return []
        except (TypeError, KeyError) as e:
            print(f"警告: 解析内部结果时出错 '{response_str[:100]}...': {e}")
            return []

    def search(self, queries: List[str]) -> List[List[Dict[str, Any]]]:
        """
        发送批量搜索请求，并处理复杂的响应格式。

        Args:
            queries: 查询列表

        Returns:
            一个列表，其中每个元素是对应查询返回的所有知识字典的列表。
            例如: [ [{'k': 'k1_1', 'c': 2.0}, {'k': 'k1_2', 'c': 1.0}], [{'k': 'k2_1', 'c': 1.5}], ... ]
        """
        request_data = {"queries": queries}
        
        try:
            response = requests.post(
                self.search_endpoint,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=300
            )
            response.raise_for_status()
            
            raw_results = response.json()
            
            if not isinstance(raw_results, list):
                print(f"错误: API响应不是一个列表，实际类型: {type(raw_results)}")
                return [[] for _ in queries]
            
            final_knowledge_list = [self._extract_all_knowledge(res_str) for res_str in raw_results]
            return final_knowledge_list

        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            return [[] for _ in queries]
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            return [[] for _ in queries]

OUTPUT_TOP_N = 5


def process_split(client: GraphR1Client, data_source: str, split_name: str, batch_size: int, top_n: int):
    """
    处理单个数据切片，获取、排序、筛选top-n知识并保存。
    """
    input_file = f'datasets/{data_source}/raw/qa_{split_name}.json'
    output_dir = f'ik_datasets/{data_source}/datasets_raw_with_top_{OUTPUT_TOP_N}_knowledge'
    output_file = os.path.join(output_dir, f'qa_{split_name}_with_top_{OUTPUT_TOP_N}_knowledge.json')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n--- processing {split_name} split (top {top_n}) ---")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 {input_file}")
        return
        
    if not data:
        print(f"警告: {input_file} 为空，跳过处理。")
        return

    print(f"成功加载 {len(data)} 条数据从 {input_file}")

    questions = [item['question'] for item in data]
    
    all_knowledge_results = []
    print(f"开始分批获取知识片段，批大小: {batch_size}")
    
    for i in tqdm(range(0, len(questions), batch_size), desc=f"Querying {split_name}"):
        batch_questions = questions[i:i + batch_size]
        batch_results = client.search(batch_questions)
        all_knowledge_results.extend(batch_results)

    print("Knowledge 获取完成。")
    
    if len(all_knowledge_results) != len(data):
        print(f"严重错误：获取到的knowledge数量 ({len(all_knowledge_results)}) 与数据量 ({len(data)}) 不匹配！")
        return

    print(f"开始排序并选择前 {top_n} 个知识片段...")

    for i, item in enumerate(data):
        # all_knowledge_results[i] 现在是一个字典列表
        # 例如: [{'knowledge': '...', 'coherence': 2.0}, ...]
        knowledge_list_with_scores = all_knowledge_results[i]
        
        # 步骤 1: 按 'coherence' 降序排序
        sorted_knowledge = sorted(
            knowledge_list_with_scores, 
            key=lambda x: x['coherence'], 
            reverse=True
        )
        
        # 步骤 2: 选择前 N 个
        top_n_knowledge = sorted_knowledge[:top_n]
        
        # 步骤 3: 只提取知识文本字符串
        final_knowledge_strings = [k['knowledge'] for k in top_n_knowledge]
        
        item['initial_knowledge'] = final_knowledge_strings
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    print(f"成功将包含 top-{top_n} 知识片段的数据保存到: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从QA数据中获取相关性最高的N个knowledge片段并保存。")
    parser.add_argument('--data_source', default='2WikiMultiHopQA', help='数据集的名称')
    parser.add_argument('--batch_size', type=int, default=64, help='每次API请求的批量大小')
    parser.add_argument('--api_url', default='http://localhost:9001', help='GraphR1 API的基础URL')
    # 新增 top_n 参数
    parser.add_argument('--top_n', type=int, default=3, help='选择相关性最高的N个知识片段')
    
    args = parser.parse_args()
    
    print("="*60)
    print("请确保您的知识检索API服务正在运行...")
    print(f"脚本将尝试连接到: {args.api_url}")
    print(f"将为每个问题检索 Top-{args.top_n} 个知识片段。")
    print("="*60)
    
    client = GraphR1Client(base_url=args.api_url)
    
    for split in ['train', 'dev', 'test']:
        # 将 top_n 参数传递给处理函数
        process_split(client, args.data_source, split, args.batch_size, args.top_n)
        
    print("\n所有数据处理完成！")


if __name__ == "__main__":
    main()
