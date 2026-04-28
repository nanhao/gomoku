"""分析 CNN v1 和 CNN v2 的参数激活情况

分析内容：
1. 各层参数量统计
2. 权重分布（均值、标准差、极值）
3. BatchNorm 统计量分析
4. ReLU 死亡神经元检测（通过权重推断）
5. 前向传播激活率分析（实际输入测试）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from ai.models.cnn_v1.model import CNNV1Network
from ai.models.cnn_v2.model import CNNV2Network


def count_parameters(model):
    """统计模型各层参数量"""
    results = []
    total_params = 0
    total_trainable = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        trainable = param.requires_grad
        results.append({
            'name': name,
            'shape': list(param.shape),
            'num_params': num_params,
            'trainable': trainable
        })
        total_params += num_params
        if trainable:
            total_trainable += num_params
    
    return results, total_params, total_trainable


def analyze_weight_distribution(model):
    """分析权重分布"""
    results = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            data = param.data.cpu().numpy()
            results.append({
                'name': name,
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'abs_mean': float(np.mean(np.abs(data))),
                'zeros_pct': float(np.sum(data == 0) / data.size * 100),
                'near_zero_pct': float(np.sum(np.abs(data) < 0.01) / data.size * 100),  # |w| < 0.01
            })
    return results


def analyze_bn_stats(model):
    """分析 BatchNorm 层的统计量"""
    results = []
    for name, buf in model.named_buffers():
        if 'running_mean' in name or 'running_var' in name:
            data = buf.cpu().numpy()
            results.append({
                'name': name,
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
            })
    return results


def analyze_fc_dead_neurons(model):
    """检测全连接层可能的死亡神经元
    
    如果 ReLU 之前的权重全部为负，该神经元永远输出 0（死亡神经元）
    对于 fc_output 之前的 ReLU (fc1)，检测 fc1 的权重行
    """
    results = []
    for name, param in model.named_parameters():
        if name == 'fc1.weight':
            data = param.data.cpu().numpy()
            num_neurons = data.shape[0]
            # 检查每个神经元的权重是否全部为负
            dead_neurons = np.all(data < 0, axis=1)
            # 检查每个神经元的权重偏置组合是否永远为负
            dead_count = int(np.sum(dead_neurons))
            
            # 更宽松的检测：权重行均值远小于0
            row_means = np.mean(data, axis=1)
            row_stds = np.std(data, axis=1)
            likely_dead = row_means + row_stds < 0  # 均值+1std < 0
            likely_dead_count = int(np.sum(likely_dead))
            
            results.append({
                'name': name,
                'total_neurons': num_neurons,
                'strictly_dead': dead_count,
                'strictly_dead_pct': dead_count / num_neurons * 100,
                'likely_dead': likely_dead_count,
                'likely_dead_pct': likely_dead_count / num_neurons * 100,
                'row_mean_of_means': float(np.mean(row_means)),
                'row_mean_of_stds': float(np.mean(row_stds)),
            })
    return results


def analyze_activation_rate(model, num_samples=100):
    """通过前向传播分析实际激活率
    
    使用随机棋盘输入，统计每层 ReLU 后的激活比例
    """
    model.eval()
    activation_stats = {}
    
    hooks = []
    
    def make_hook(layer_name):
        def hook(module, input, output):
            if layer_name not in activation_stats:
                activation_stats[layer_name] = []
            # output 是 ReLU 的输出
            data = output.detach().cpu().numpy()
            active_rate = float(np.mean(data > 0))  # 激活比例（>0）
            mean_active = float(np.mean(data[data > 0])) if np.any(data > 0) else 0.0
            activation_stats[layer_name].append({
                'active_rate': active_rate,
                'mean_active_value': mean_active,
                'output_shape': list(data.shape),
            })
        return hook
    
    # 注册 hook
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # 生成随机棋盘输入
    with torch.no_grad():
        for _ in range(num_samples):
            # 模拟随机棋盘状态
            board = torch.zeros(1, 3, 15, 15)
            # 通道1: 黑棋位置（随机）
            board[0, 0] = (torch.rand(15, 15) > 0.7).float()
            # 通道2: 白棋位置（随机）
            board[0, 1] = (torch.rand(15, 15) > 0.7).float()
            # 通道3: 当前方
            board[0, 2] = 1.0
            
            model(board)
    
    # 移除 hooks
    for h in hooks:
        h.remove()
    
    # 汇总统计
    summary = {}
    for layer_name, stats_list in activation_stats.items():
        active_rates = [s['active_rate'] for s in stats_list]
        mean_values = [s['mean_active_value'] for s in stats_list]
        summary[layer_name] = {
            'avg_active_rate': float(np.mean(active_rates)),
            'std_active_rate': float(np.std(active_rates)),
            'min_active_rate': float(np.min(active_rates)),
            'max_active_rate': float(np.max(active_rates)),
            'avg_mean_active_value': float(np.mean(mean_values)),
        }
    
    return summary


def print_separator(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_table(headers, rows):
    """打印格式化表格"""
    col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    
    header_line = " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    sep_line = "-+-".join("-" * w for w in col_widths)
    
    print(header_line)
    print(sep_line)
    for row in rows:
        line = " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        print(line)


def main():
    device = torch.device('cpu')
    
    # 加载模型
    cnn_v1 = CNNV1Network()
    v1_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'ai', 'models', 'cnn_v1', 'cnn.pth')
    cnn_v1.load_state_dict(torch.load(v1_path, map_location=device, weights_only=True))
    cnn_v1.eval()
    
    cnn_v2 = CNNV2Network()
    v2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'ai', 'models', 'cnn_v2', 'cnn_v2.pth')
    cnn_v2.load_state_dict(torch.load(v2_path, map_location=device, weights_only=True))
    cnn_v2.eval()
    
    models = {'CNN v1': cnn_v1, 'CNN v2': cnn_v2}
    
    # ========== 1. 参数量统计 ==========
    print_separator("1. 各层参数量统计")
    
    for model_name, model in models.items():
        params, total, trainable = count_parameters(model)
        print(f"\n--- {model_name} (总计: {total:,}, 可训练: {trainable:,}) ---")
        headers = ["层名称", "形状", "参数量", "占比%"]
        rows = []
        for p in params:
            pct = p['num_params'] / total * 100
            rows.append([p['name'], str(p['shape']), f"{p['num_params']:,}", f"{pct:.1f}%"])
        print_table(headers, rows)
    
    # 对比总结
    _, v1_total, _ = count_parameters(cnn_v1)
    _, v2_total, _ = count_parameters(cnn_v2)
    print(f"\n参数量对比: CNN v1 = {v1_total:,}, CNN v2 = {v2_total:,}, "
          f"减少 {(v1_total - v2_total) / v1_total * 100:.1f}%")
    
    # ========== 2. 权重分布分析 ==========
    print_separator("2. 权重分布分析")
    
    for model_name, model in models.items():
        stats = analyze_weight_distribution(model)
        print(f"\n--- {model_name} ---")
        headers = ["层名称", "均值", "标准差", "最小值", "最大值", "|w|均值", "近零%"]
        rows = []
        for s in stats:
            rows.append([
                s['name'],
                f"{s['mean']:.6f}",
                f"{s['std']:.6f}",
                f"{s['min']:.4f}",
                f"{s['max']:.4f}",
                f"{s['abs_mean']:.6f}",
                f"{s['near_zero_pct']:.1f}%",
            ])
        print_table(headers, rows)
    
    # ========== 3. BatchNorm 统计量分析 ==========
    print_separator("3. BatchNorm 统计量分析")
    
    for model_name, model in models.items():
        stats = analyze_bn_stats(model)
        print(f"\n--- {model_name} ---")
        headers = ["名称", "均值", "标准差", "最小值", "最大值"]
        rows = []
        for s in stats:
            rows.append([
                s['name'],
                f"{s['mean']:.6f}",
                f"{s['std']:.6f}",
                f"{s['min']:.6f}",
                f"{s['max']:.6f}",
            ])
        print_table(headers, rows)
    
    # ========== 4. 全连接层死亡神经元检测 ==========
    print_separator("4. 全连接层死亡神经元检测 (fc1)")
    
    for model_name, model in models.items():
        stats = analyze_fc_dead_neurons(model)
        print(f"\n--- {model_name} ---")
        for s in stats:
            print(f"  层: {s['name']}")
            print(f"  总神经元数: {s['total_neurons']}")
            print(f"  严格死亡(全负权重): {s['strictly_dead']} ({s['strictly_dead_pct']:.1f}%)")
            print(f"  可能死亡(均值+1σ<0): {s['likely_dead']} ({s['likely_dead_pct']:.1f}%)")
            print(f"  权重行均值: {s['row_mean_of_means']:.6f}")
            print(f"  权重行标准差: {s['row_mean_of_stds']:.6f}")
    
    # ========== 5. 前向传播激活率分析 ==========
    print_separator("5. 前向传播激活率分析 (100次随机输入)")
    
    for model_name, model in models.items():
        summary = analyze_activation_rate(model, num_samples=100)
        print(f"\n--- {model_name} ---")
        headers = ["ReLU层", "平均激活率", "激活率标准差", "最小激活率", "最大激活率", "平均激活值"]
        rows = []
        for layer_name, stats in summary.items():
            rows.append([
                layer_name,
                f"{stats['avg_active_rate']:.4f}",
                f"{stats['std_active_rate']:.4f}",
                f"{stats['min_active_rate']:.4f}",
                f"{stats['max_active_rate']:.4f}",
                f"{stats['avg_mean_active_value']:.4f}",
            ])
        print_table(headers, rows)
    
    # ========== 6. 综合对比 ==========
    print_separator("6. 综合对比总结")
    
    v1_params, _, _ = count_parameters(cnn_v1)
    v2_params, _, _ = count_parameters(cnn_v2)
    
    v1_conv_params = sum(p['num_params'] for p in v1_params if 'conv' in p['name'])
    v2_conv_params = sum(p['num_params'] for p in v2_params if 'conv' in p['name'])
    
    v1_fc_params = sum(p['num_params'] for p in v1_params if 'fc' in p['name'])
    v2_fc_params = sum(p['num_params'] for p in v2_params if 'fc' in p['name'])
    
    v1_bn_params = sum(p['num_params'] for p in v1_params if 'bn' in p['name'])
    v2_bn_params = sum(p['num_params'] for p in v2_params if 'bn' in p['name'])
    
    print(f"\n{'指标':<20} {'CNN v1':>15} {'CNN v2':>15} {'变化':>15}")
    print("-" * 70)
    print(f"{'总参数量':<20} {v1_total:>15,} {v2_total:>15,} {(v2_total-v1_total)/v1_total*100:>+14.1f}%")
    print(f"{'卷积层参数':<20} {v1_conv_params:>15,} {v2_conv_params:>15,} {(v2_conv_params-v1_conv_params)/v1_conv_params*100:>+14.1f}%")
    print(f"{'全连接层参数':<20} {v1_fc_params:>15,} {v2_fc_params:>15,} {(v2_fc_params-v1_fc_params)/v1_fc_params*100:>+14.1f}%")
    print(f"{'BN层参数':<20} {v1_bn_params:>15,} {v2_bn_params:>15,} {(v2_bn_params-v1_bn_params)/v1_bn_params*100:>+14.1f}%")
    
    # FC1 参数占比
    v1_fc1 = sum(p['num_params'] for p in v1_params if p['name'] in ('fc1.weight', 'fc1.bias'))
    v2_fc1 = sum(p['num_params'] for p in v2_params if p['name'] in ('fc1.weight', 'fc1.bias'))
    print(f"\n{'FC1 参数占比':<20} {v1_fc1/v1_total*100:>14.1f}% {v2_fc1/v2_total*100:>14.1f}%")
    
    print("\n分析完成！")


if __name__ == '__main__':
    main()
