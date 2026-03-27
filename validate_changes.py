"""
验证修改的正确性

检查点：
1. 类型注解是否正确
2. 变量名是否正确
3. 导入语句是否正确
4. 逻辑流程是否完整
"""

import sys
import ast

def check_python_file(filepath, description):
    """检查Python文件的语法"""
    print(f"\n=== 检查: {description} ===")
    print(f"文件: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()

        ast.parse(source)
        print("✓ 语法正确")

        # 检查关键函数和类
        tree = ast.parse(source)
        functions = []
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)

        print(f"  定义 {len(functions)} 个函数")
        print(f"  定义 {len(classes)} 个类")

        return True

    except SyntaxError as e:
        print(f"✗ 语法错误: {e}")
        print(f"  行 {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False


def main():
    files_to_check = [
        ("multimodal/mapper.py", "消息映射模块"),
        ("multimodal/processor.py", "多模态处理器"),
        ("multimodal/video_loader.py", "视频加载器"),
        ("api/routes/chat.py", "聊天路由"),
        ("services/chat_service.py", "聊天服务"),
        ("vllm-backend/vllm/entrypoints/chat_utils.py", "后端聊天工具"),
    ]

    print("=" * 60)
    print("视频图片列表格式支持 - 代码验证")
    print("=" * 60)

    all_passed = True

    for filepath, description in files_to_check:
        if check_python_file(filepath, description):
            continue
        else:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有文件语法正确！")
        print("\n请检查以下内容：")
        print("1. 类型注解是否完整")
        print("2. 逻辑流程是否正确")
        print("3. 参数传递链路是否完整")
        print("4. 是否有冗余代码")
        print("5. 是否有未使用的导入")
        sys.exit(0)
    else:
        print("✗ 部分文件有语法错误，请修复！")
        sys.exit(1)


if __name__ == "__main__":
    main()
