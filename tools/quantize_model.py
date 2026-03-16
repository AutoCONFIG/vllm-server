#!/usr/bin/env python3
"""
模型量化工具

支持多种量化方法:
- AutoAWQ: AWQ (Activation-aware Weight Quantization)
- GPTQModel: GPTQ 量化
- LLM Compressor: FP8, INT4, INT8 等多种格式

用法:
    # AWQ 量化
    python tools/quantize_model.py --method awq --input-model /path/to/model --output-model /path/to/output --bits 4

    # GPTQ 量化
    python tools/quantize_model.py --method gptq --input-model /path/to/model --output-model /path/to/output --bits 4

    # FP8 量化
    python tools/quantize_model.py --method fp8 --input-model /path/to/model --output-model /path/to/output
"""

import argparse
import os
import sys
from pathlib import Path


def quantize_awq(
    input_model: str,
    output_model: str,
    bits: int = 4,
    group_size: int = 128,
    zero_point: bool = True,
    version: str = "gemm",
):
    """使用 AutoAWQ 进行量化"""
    try:
        from awq import AutoAWQ
        from transformers import AutoTokenizer
    except ImportError:
        print("错误: 请先安装 AutoAWQ")
        print("运行: pip install autoawq")
        sys.exit(1)

    print(f"🔄 正在使用 AWQ 量化模型...")
    print(f"   输入模型: {input_model}")
    print(f"   输出模型: {output_model}")
    print(f"   量化位数: {bits}-bit")
    print(f"   分组大小: {group_size}")

    # 加载模型和分词器
    print("📦 正在加载模型...")
    quantizer = AutoAWQ.from_pretrained(input_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(input_model, trust_remote_code=True)

    # 执行量化
    print("⚙️  正在量化...")
    quantizer.quantize(
        quant_path=output_model,
        quant_method="awq",
        bits=bits,
        group_size=group_size,
        zero_point=zero_point,
        version=version,
    )

    # 保存分词器
    tokenizer.save_pretrained(output_model)

    print(f"✅ AWQ 量化完成！")
    print(f"   模型保存到: {output_model}")
    print(f"\n📝 在 vLLM 中使用的配置:")
    print(f"   model:")
    print(f"     path: \"{output_model}\"")
    print(f"     dtype: \"float16\"")
    print(f"     quantization: \"awq\"")


def quantize_gptq(
    input_model: str,
    output_model: str,
    bits: int = 4,
    group_size: int = 128,
):
    """使用 GPTQModel 进行量化"""
    try:
        from gptqmodel import GPTQModel, quantization
    except ImportError:
        print("错误: 请先安装 GPTQModel")
        print("运行: pip install gptqmodel")
        sys.exit(1)

    print(f"🔄 正在使用 GPTQ 量化模型...")
    print(f"   输入模型: {input_model}")
    print(f"   输出模型: {output_model}")
    print(f"   量化位数: {bits}-bit")
    print(f"   分组大小: {group_size}")

    # 加载模型
    print("📦 正在加载模型...")
    model = GPTQModel.load(
        input_model,
        trust_remote_code=True,
    )

    # 执行量化
    print("⚙️  正在量化...")
    quantization.configure(bits=bits, group_size=group_size)
    model.quantize(output_model)

    print(f"✅ GPTQ 量化完成！")
    print(f"   模型保存到: {output_model}")
    print(f"\n📝 在 vLLM 中使用的配置:")
    print(f"   model:")
    print(f"     path: \"{output_model}\"")
    print(f"     dtype: \"float16\"")
    print(f"     quantization: \"gptq\"")


def quantize_fp8(
    input_model: str,
    output_model: str,
):
    """使用 LLM Compressor 进行 FP8 量化"""
    try:
        from llmcompressor import oneshot
        from llmcompressor.transformers import AutoModelForCausalLM
    except ImportError:
        print("错误: 请先安装 llm-compressor")
        print("运行: pip install llm-compressor")
        sys.exit(1)

    print(f"🔄 正在使用 FP8 量化模型...")
    print(f"   输入模型: {input_model}")
    print(f"   输出模型: {output_model}")

    # 加载模型
    print("📦 正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        input_model,
        device_map="auto",
        trust_remote_code=True,
    )

    # 执行 FP8 量化
    print("⚙️  正在量化...")
    oneshot(
        model=model,
        recipe="experimental::fp8",
        output_model=output_model,
    )

    print(f"✅ FP8 量化完成！")
    print(f"   模型保存到: {output_model}")
    print(f"\n📝 在 vLLM 中使用的配置:")
    print(f"   model:")
    print(f"     path: \"{output_model}\"")
    print(f"     dtype: \"auto\"")
    print(f"     quantization: \"fp8\"")


def quantize_int4_llm_compressor(
    input_model: str,
    output_model: str,
    bits: int = 4,
    group_size: int = 128,
):
    """使用 LLM Compressor 进行 INT4 量化"""
    try:
        from llmcompressor import oneshot
        from llmcompressor.transformers import AutoModelForCausalLM
    except ImportError:
        print("错误: 请先安装 llm-compressor")
        print("运行: pip install llm-compressor")
        sys.exit(1)

    print(f"🔄 正在使用 INT4 量化模型...")
    print(f"   输入模型: {input_model}")
    print(f"   输出模型: {output_model}")
    print(f"   量化位数: {bits}-bit")
    print(f"   分组大小: {group_size}")

    # 加载模型
    print("📦 正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        input_model,
        device_map="auto",
        trust_remote_code=True,
    )

    # 执行 INT4 量化
    print("⚙️  正在量化...")
    oneshot(
        model=model,
        recipe=f"quantization::W{bits}A16",
        output_model=output_model,
    )

    print(f"✅ INT4 量化完成！")
    print(f"   模型保存到: {output_model}")
    print(f"\n📝 在 vLLM 中使用的配置:")
    print(f"   model:")
    print(f"     path: \"{output_model}\"")
    print(f"     dtype: \"float16\"")
    print(f"     quantization: \"compressed-tensors\"")


def main():
    parser = argparse.ArgumentParser(
        description="模型量化工具 - 支持 AWQ, GPTQ, FP8, INT4 等量化方法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # AWQ 量化 (推荐用于 INT4)
  python tools/quantize_model.py --method awq \\
    --input-model meta-llama/Llama-2-7b-hf \\
    --output-model ./llama-2-7b-awq \\
    --bits 4

  # GPTQ 量化
  python tools/quantize_model.py --method gptq \\
    --input-model meta-llama/Llama-2-7b-hf \\
    --output-model ./llama-2-7b-gptq \\
    --bits 4

  # FP8 量化
  python tools/quantize_model.py --method fp8 \\
    --input-model meta-llama/Llama-2-7b-hf \\
    --output-model ./llama-2-7b-fp8

  # INT4 量化 (使用 llm-compressor)
  python tools/quantize_model.py --method int4 \\
    --input-model meta-llama/Llama-2-7b-hf \\
    --output-model ./llama-2-7b-int4 \\
    --bits 4
        """
    )

    parser.add_argument(
        "--method", "-m",
        type=str,
        required=True,
        choices=["awq", "gptq", "fp8", "int4"],
        help="量化方法: awq (INT4), gptq (INT4), fp8 (FP8), int4 (INT4)"
    )

    parser.add_argument(
        "--input-model", "-i",
        type=str,
        required=True,
        help="输入模型路径 (Hugging Face 模型ID或本地路径)"
    )

    parser.add_argument(
        "--output-model", "-o",
        type=str,
        required=True,
        help="输出模型路径"
    )

    parser.add_argument(
        "--bits", "-b",
        type=int,
        default=4,
        help="量化位数 (默认: 4)"
    )

    parser.add_argument(
        "--group-size", "-g",
        type=int,
        default=128,
        help="分组大小 (默认: 128)"
    )

    parser.add_argument(
        "--zero-point",
        action="store_true",
        default=True,
        help="使用零点量化 (默认: True)"
    )

    parser.add_argument(
        "--awq-version",
        type=str,
        default="gemm",
        choices=["gemm", "gemv"],
        help="AWQ 实现版本 (默认: gemm)"
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_model, exist_ok=True)

    # 根据量化方法调用对应函数
    if args.method == "awq":
        quantize_awq(
            input_model=args.input_model,
            output_model=args.output_model,
            bits=args.bits,
            group_size=args.group_size,
            zero_point=args.zero_point,
            version=args.awq_version,
        )
    elif args.method == "gptq":
        quantize_gptq(
            input_model=args.input_model,
            output_model=args.output_model,
            bits=args.bits,
            group_size=args.group_size,
        )
    elif args.method == "fp8":
        quantize_fp8(
            input_model=args.input_model,
            output_model=args.output_model,
        )
    elif args.method == "int4":
        quantize_int4_llm_compressor(
            input_model=args.input_model,
            output_model=args.output_model,
            bits=args.bits,
            group_size=args.group_size,
        )


if __name__ == "__main__":
    main()
