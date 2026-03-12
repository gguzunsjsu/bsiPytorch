#!/usr/bin/env python3
import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit('matplotlib is required to generate the benchmark figures') from exc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'figures' / 'generated'
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def save_memory_plot(rows):
    models = [r['model'] for r in rows]
    dense = [float(r['dense_mb']) for r in rows]
    all_mb = [float(r['bsi_all_mb']) for r in rows]
    attn_mb = [float(r['bsi_attention_mb']) for r in rows]

    x = range(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width for i in x], dense, width=width, label='Dense FP16')
    ax.bar(list(x), all_mb, width=width, label='BSI scope=all')
    ax.bar([i + width for i in x], attn_mb, width=width, label='BSI scope=attention')
    ax.set_xticks(list(x), models)
    ax.set_ylabel('Static model size (MB)')
    ax.set_title('Static model size across dense and BSI configurations')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'memory_comparison.png', dpi=200)
    plt.close(fig)


def save_latency_plot(rows):
    models = [r['model'] for r in rows]
    bsi_all = [float(r['bsi_all_dot_ms']) for r in rows]
    torch_all = [float(r['torch_all_dot_ms']) for r in rows]
    bsi_attn = [float(r['bsi_attention_dot_ms']) for r in rows]
    torch_attn = [float(r['torch_attention_dot_ms']) for r in rows]

    x = range(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - 1.5 * width for i in x], bsi_all, width=width, label='BSI scope=all')
    ax.bar([i - 0.5 * width for i in x], torch_all, width=width, label='Torch scope=all')
    ax.bar([i + 0.5 * width for i in x], bsi_attn, width=width, label='BSI scope=attention')
    ax.bar([i + 1.5 * width for i in x], torch_attn, width=width, label='Torch scope=attention')
    ax.set_xticks(list(x), models)
    ax.set_ylabel('Per-sample dot latency (ms)')
    ax.set_title('Dot latency across dense and BSI configurations')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'dot_latency_comparison.png', dpi=200)
    plt.close(fig)


def save_accuracy_plot(path, out_name, title):
    rows = read_csv(path)
    models = [r['model'] for r in rows]
    dense_top1 = [float(r['dense_top1']) for r in rows]
    bsi_top1 = [float(r['bsi_top1']) for r in rows]
    dense_top5 = [float(r['dense_top5']) for r in rows]
    bsi_top5 = [float(r['bsi_top5']) for r in rows]

    x = range(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - 1.5 * width for i in x], dense_top1, width=width, label='Dense Top-1')
    ax.bar([i - 0.5 * width for i in x], bsi_top1, width=width, label='BSI Top-1')
    ax.bar([i + 0.5 * width for i in x], dense_top5, width=width, label='Dense Top-5')
    ax.bar([i + 1.5 * width for i in x], bsi_top5, width=width, label='BSI Top-5')
    ax.set_xticks(list(x), models)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / out_name, dpi=200)
    plt.close(fig)


def main():
    memory_rows = read_csv(DATA / 'memory_dot_latency.csv')
    save_memory_plot(memory_rows)
    save_latency_plot(memory_rows)
    save_accuracy_plot(DATA / 'accuracy_scope_all.csv', 'accuracy_scope_all.png', 'Accuracy under scope=all')
    save_accuracy_plot(DATA / 'accuracy_scope_attention.csv', 'accuracy_scope_attention.png', 'Accuracy under scope=attention')
    print(f'Wrote figures to {OUT}')


if __name__ == '__main__':
    main()
