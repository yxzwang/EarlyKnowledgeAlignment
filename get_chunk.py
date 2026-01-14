import os
import json
import argparse
import requests
from typing import List, Dict, Any

# ==============================================================================
#  复用您提供的 GraphR1Client 类
# ==============================================================================

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
            
            knowledge_with_coherence = []
            for item in results_list:
                if '<knowledge>' in item and '<coherence>' in item:
                    try:
                        knowledge_with_coherence.append({
                            'knowledge': item.get('<knowledge>', ''),
                            'coherence': float(item.get('<coherence>'))
                        })
                    except (ValueError, TypeError):
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

# ==============================================================================
#  新的脚本逻辑：检索所有知识并存入JSONL文件
# ==============================================================================

def retrieve_and_save_all(client: GraphR1Client, query: str, output_file: str):
    """
    为单个查询检索所有知识，并按指定格式保存到JSONL文件中。

    Args:
        client: GraphR1Client的实例。
        query: 用于检索的查询字符串。
        output_file: 输出的JSONL文件名。
    """
    print(f"\n--- 使用以下查询进行检索 ---")
    print(f"查询: '{query}'")
    
    # client.search 期望一个查询列表，我们只传递一个查询。
    # API将返回一个结果列表，其中包含我们这一个查询的所有结果。
    all_results = client.search([query])

    # 检查API调用是否成功并且返回了数据
    if not all_results or not all_results[0]:
        print("未检索到任何知识，或者API返回了错误/空结果。")
        print(f"无法生成文件 {output_file}。")
        return

    # all_results[0] 包含了我们查询的所有知识片段（一个字典列表）
    knowledge_snippets = all_results[0]
    
    print(f"成功检索到 {len(knowledge_snippets)} 条知识片段。")
    print(f"正在将结果写入到: {output_file}")

    # 以写入模式打开文件，准备写入JSONL数据
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # 遍历所有检索到的知识片段
            for i, snippet_dict in enumerate(knowledge_snippets):
                # 根据要求构建新的字典
                # "id" 是从0开始的字符串索引
                # "contents" 是知识文本
                output_record = {
                    "id": str(i),
                    "contents": snippet_dict.get('knowledge', '')  # 使用.get()确保安全
                }
                
                # 将字典转换为JSON字符串，并添加换行符，形成JSONL格式
                f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
        
        print(f"\n处理完成！所有 {len(knowledge_snippets)} 条内容已成功保存到 {output_file}")

    except IOError as e:
        print(f"错误: 无法写入文件 {output_file}。原因: {e}")


def main():
    """主函数，用于解析命令行参数并启动检索过程。"""
    parser = argparse.ArgumentParser(
        description="使用单个查询从GraphR1 API检索所有知识，并将其保存为JSONL文件。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--query', 
        type=str,
        default="123"
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='retrieved_knowledge_triviaqa.jsonl',
        help='用于保存结果的输出文件名。\n默认为: retrieved_knowledge.jsonl'
    )
    parser.add_argument(
        '--api_url', 
        default='http://localhost:9001', 
        help='GraphR1 API的基础URL。\n默认为: http://localhost:9001'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("请确保您的知识检索API服务正在运行...")
    print(f"脚本将尝试连接到: {args.api_url}")
    print("="*60)
    
    # 1. 初始化API客户端
    client = GraphR1Client(base_url=args.api_url)
    
    # 2. 执行检索和保存操作
    retrieve_and_save_all(client, args.query, args.output_file)


if __name__ == "__main__":
    main()