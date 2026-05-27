import json, os, sys
sys.path.insert(0, '.')

# 找一个有 sample.json 的对局
rec_path = None
sf_path = None
for root, dirs, files in os.walk('ai/train_data'):
    for f in files:
        if f == 'sample.json':
            sf = os.path.join(root, f)
            rec = sf.replace('train_data', 'ai_data').replace('sample.json', 'record.json')
            if os.path.exists(rec):
                rec_path = rec
                sf_path = sf
                break
    if rec_path:
        break

r = json.load(open(rec_path))
s = json.load(open(sf_path))

print(f'record: {rec_path}')
print(f'sample: {sf_path}')
print(f'history steps: {len(r["history"])}')
print(f'sample count: {len(s)}')
print(f'winner: {r.get("winner")}')
print()

# 关键验证：sample[i]["output"] 是否等于 Evaluator 算出来的值？
# Evaluator 使用 RewardCalculator + masked_softmax
# 而 SampleCreator 使用 _build_winner_path_output (基于 AI output)
# 如果 sample[i]["output"] 和 AI output 差异很大，说明它来自 Evaluator

from ai.evaluator import Evaluator
ev = Evaluator()
sample_computed = ev._compute_all_rewards(r['history'], r.get('winner',0), r.get('end_board',[]))

# 比较前3步
for i in range(min(3, len(s))):
    # 比较是否与 Evaluator 输出一致
    match_eval = all(abs(s[i]['output'][j] - sample_computed[i]['output'][j]) < 1e-10 for j in range(225))
    # 比较是否与 AI output 一致（SampleCreator soft label 应该与 AI output 接近）
    ai_out = r['history'][i]['output']
    corr_with_ai = sum(1 for j in range(225) if abs(s[i]['output'][j] - ai_out[j]) < 1e-6)
    
    print(f'step {i+1}:')
    print(f'  match Evaluator: {match_eval}')
    print(f'  correlation with AI output: {corr_with_ai}/225 positions match')
    print(f'  AI max={max(ai_out):.4f}  Sample max={max(s[i]["output"]):.4f}  Evaluator max={max(sample_computed[i]["output"]):.4f}')
    print()
