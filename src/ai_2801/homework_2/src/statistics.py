"""
统计和分析语言模型验证结果
遍历 result 目录下所有模型的验证结果，计算统计指标并可视化
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List


def safe_float_convert(value, default=None):
    """
    安全地将值转换为浮点数

    Args:
        value: 要转换的值
        default: 转换失败时的默认值

    Returns:
        浮点数或默认值
    """
    if value is None:
        return default

    try:
        float_val = float(value)
        # 检查是否为有效数值（非 nan, 非 inf）
        if np.isfinite(float_val):
            return float_val
        else:
            return default
    except (ValueError, TypeError):
        return default


def load_results(result_dir: str) -> Dict[str, List[dict]]:
    """
    加载所有模型的验证结果

    Args:
        result_dir: 结果目录路径

    Returns:
        字典，key 为模型名称，value 为该模型的所有结果列表
    """
    results = {}

    for model_dir in Path(result_dir).iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            result_file = model_dir / "results.jsonl"

            if result_file.exists():
                model_results = []
                invalid_count = 0

                with open(result_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line.strip())

                            # 安全转换 PPL 为浮点数
                            if 'ppl' in data:
                                ppl_value = safe_float_convert(data['ppl'])
                                if ppl_value is not None:
                                    data['ppl'] = ppl_value
                                    model_results.append(data)
                                else:
                                    invalid_count += 1
                                    if invalid_count <= 5:  # 只打印前5个警告
                                        print(f"Warning: Invalid PPL value at line {line_num}: {data.get('ppl')}")
                            else:
                                invalid_count += 1

                        except json.JSONDecodeError as e:
                            invalid_count += 1
                            if invalid_count <= 5:
                                print(f"Warning: JSON decode error at line {line_num}: {e}")
                            continue

                if invalid_count > 5:
                    print(f"Warning: {invalid_count - 5} more invalid entries found for {model_name}")

                results[model_name] = model_results
                print(f"Loaded {len(model_results)} valid results for {model_name} (filtered {invalid_count} invalid)")

    return results


def filter_outliers(ppls_array: np.ndarray, outlier_percentile: float = 90) -> tuple:
    """
    筛除异常值，移除最大的 outlier_percentile% 的数据

    Args:
        ppls_array: PPL 数值数组
        outlier_percentile: 异常值阈值百分位（默认 90，即移除最大的 10%）

    Returns:
        (过滤后的数组, 被过滤掉的数组)
    """
    if len(ppls_array) == 0:
        return ppls_array, np.array([])

    threshold = np.percentile(ppls_array, outlier_percentile)
    filtered = ppls_array[ppls_array <= threshold]
    outliers = ppls_array[ppls_array > threshold]

    return filtered, outliers


def calculate_statistics(results: Dict[str, List[dict]]) -> pd.DataFrame:
    """
    计算统计指标：均值、方差等
    自动筛除最大的 10% 异常值

    Args:
        results: 模型结果字典

    Returns:
        包含统计信息的 DataFrame
    """
    stats = []

    for model_name, model_results in results.items():
        # 提取 PPL 值（确保都是有效的浮点数）
        ppls = []
        for r in model_results:
            ppl_value = safe_float_convert(r.get('ppl'))
            if ppl_value is not None:
                ppls.append(ppl_value)

        if ppls:
            ppls_array = np.array(ppls)

            # 额外验证：确保没有 nan 或 inf
            if not np.all(np.isfinite(ppls_array)):
                print(f"Warning: Found non-finite values in {model_name}, filtering them out")
                ppls_array = ppls_array[np.isfinite(ppls_array)]

            # 筛除最大的 10% 异常值
            filtered_ppls, outliers = filter_outliers(ppls_array, outlier_percentile=90)

            if len(outliers) > 0:
                print(f"{model_name}: Filtered {len(outliers)} outliers ({len(outliers)/len(ppls_array)*100:.1f}%)")
                print(f"  Outlier range: [{np.min(outliers):.2f}, {np.max(outliers):.2f}]")

            if len(filtered_ppls) > 0:
                stats.append({
                    'model': model_name,
                    'total_count': len(ppls_array),
                    'filtered_count': len(filtered_ppls),
                    'outliers_count': len(outliers),
                    'mean': np.mean(filtered_ppls),
                    'std': np.std(filtered_ppls),
                    'var': np.var(filtered_ppls),
                    'median': np.median(filtered_ppls),
                    'min': np.min(filtered_ppls),
                    'max': np.max(filtered_ppls),
                    'q25': np.percentile(filtered_ppls, 25),
                    'q75': np.percentile(filtered_ppls, 75),
                })
            else:
                print(f"Warning: No valid PPL values found for {model_name} after filtering")

    df = pd.DataFrame(stats)
    return df


def plot_ppl_distributions(results: Dict[str, List[dict]], output_dir: str):
    """
    绘制 PPL 分布图并保存为 PDF
    只生成直方图对比图

    Args:
        results: 模型结果字典
        output_dir: 图像输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # 准备数据并筛除异常值
    model_names = []
    ppls_data = []

    for model_name, model_results in results.items():
        # 安全提取和验证 PPL 值
        ppls = []
        for r in model_results:
            ppl_value = safe_float_convert(r.get('ppl'))
            if ppl_value is not None:
                ppls.append(ppl_value)

        if ppls:
            # 转换为 numpy 数组并验证
            ppls_array = np.array(ppls)
            ppls_array = ppls_array[np.isfinite(ppls_array)]  # 过滤 nan/inf

            # 筛除异常值
            filtered_ppls, outliers = filter_outliers(ppls_array, outlier_percentile=90)

            if len(filtered_ppls) > 0:
                model_names.append(model_name)
                ppls_data.append(filtered_ppls.tolist())
                print(f"{model_name}: Using {len(filtered_ppls)} samples (filtered {len(outliers)} outliers)")
            else:
                print(f"Warning: No valid PPL data for {model_name} after filtering")

    if not ppls_data:
        print("No valid PPL data found for plotting")
        return

    # 创建直方图对比图
    fig, ax = plt.subplots(figsize=(12, 8))

    for model_name, ppls in zip(model_names, ppls_data):
        ax.hist(ppls, bins=50, alpha=0.6, label=model_name, density=True)

    ax.set_xlabel('Perplexity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('PPL Comparison on PTB.Test', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    output_path = os.path.join(output_dir, 'ppl_distribution_comparison.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved histogram comparison plot to {output_path}")
    plt.close()


def generate_statistics_report(results: Dict[str, List[dict]], output_dir: str):
    """
    生成统计报告并保存

    Args:
        results: 模型结果字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 计算统计信息
    df_stats = calculate_statistics(results)

    # 保存为 CSV
    csv_path = os.path.join(output_dir, 'statistics_summary.csv')
    df_stats.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Saved statistics summary to {csv_path}")

    # 保存为 Markdown
    md_path = os.path.join(output_dir, 'statistics_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# Language Model Validation Statistics\n\n')
        f.write('## Summary Table\n\n')
        f.write(df_stats.to_markdown(index=False, floatfmt='.4f'))
        f.write('\n\n## Detailed Analysis\n\n')

        for _, row in df_stats.iterrows():
            model_name = row['model']
            f.write(f'### {model_name}\n\n')
            f.write(f'- **Total Samples**: {row["total_count"]}\n')
            f.write(f'- **Filtered Samples**: {row["filtered_count"]} ({row["filtered_count"]/row["total_count"]*100:.1f}%)\n')
            f.write(f'- **Outliers Removed**: {row["outliers_count"]} ({row["outliers_count"]/row["total_count"]*100:.1f}%)\n')
            f.write(f'- **Mean PPL** (filtered): {row["mean"]:.4f}\n')
            f.write(f'- **Std Dev**: {row["std"]:.4f}\n')
            f.write(f'- **Variance**: {row["var"]:.4f}\n')
            f.write(f'- **Median**: {row["median"]:.4f}\n')
            f.write(f'- **Range** (filtered): [{row["min"]:.4f}, {row["max"]:.4f}]\n')
            f.write(f'- **IQR**: [{row["q25"]:.4f}, {row["q75"]:.4f}]\n\n')

    print(f"Saved markdown report to {md_path}")

    # 打印到控制台
    print("\n" + "="*80)
    print("STATISTICS SUMMARY")
    print("="*80)
    print(df_stats.to_string(index=False))
    print("="*80)

    return df_stats


def main():
    """主函数"""
    # 设置路径
    result_dir = "src/ai_2801/homework_2/result"
    output_dir = "/data/xiyuanyang/My-NLP/src/ai_2801/homework_2/images"

    print("Loading validation results...")
    results = load_results(result_dir)

    if not results:
        print("No results found in directory!")
        return

    print(f"\nFound results for {len(results)} models")

    # 生成统计报告
    print("\nGenerating statistics report...")
    generate_statistics_report(results, output_dir)

    # 绘制分布图
    print("\nPlotting distributions...")
    plot_ppl_distributions(results, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
