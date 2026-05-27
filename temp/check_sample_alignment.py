import json, os

# 找一个 winner_path 模式的 sample.json（条数 ≈ history/2）
for root, dirs, files in os.walk('ai/train_data'):
    for f in files:
        if f == 'sample.json':
            sf = os.path.join(root, f)
            rec = sf.replace('train_data', 'ai_data').replace('sample.json', 'record.json')
            if os.path.exists(rec):
                r = json.load(open(rec))
                s = json.load(open(sf))
                h = len(r['history'])
                sc = len(s)
                w = r.get('winner')
                # 只看 sample 数 != history 数的（winner_path 模式）
                if sc != h:
                    print(f'FOUND mismatch: {rec}')
                    print(f'history={h} samples={sc} winner={w}')
                    # 看每条 sample 对应 history 的第几步
                    for si in range(min(5, sc)):
                        # 用 input 匹配：sample 的 input 应与某步 history 的 input 完全相同
                        for hi in range(h):
                            if s[si]['input'] == r['history'][hi]['input']:
                                print(f'  sample[{si}] matches history[{hi}] (step {hi+1}), pos={r["history"][hi]["position"]}')
                                break
                    print()
                    break
    else:
        continue
    break
else:
    print('No mismatched sample files found. All have sample_count == history_count.')
