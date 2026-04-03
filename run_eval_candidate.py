import sys
import json
import traceback

# Ensure we can import the tuner from Python/ folder
sys.path.insert(0, 'Python')
try:
    from reward_weight_tuner import evaluate
except Exception:
    print('ERROR: cannot import reward_weight_tuner; check PYTHONPATH')
    traceback.print_exc()
    sys.exit(2)

cfg_file = 'eval_candidate.json'
try:
    with open(cfg_file, 'r', encoding='utf-8') as f:
        weights = json.load(f)
except Exception as e:
    print('ERROR: failed to read', cfg_file, e)
    traceback.print_exc()
    sys.exit(3)

print('Starting evaluate...')
try:
    # train_rl.py 位于 Python/ 目录，传入该目录作为 code_dir
    score = evaluate(weights, 'Python', 1, 20, 600)
    print('EVAL_SCORE:', score)
except Exception as e:
    print('EVAL_ERROR:', e)
    traceback.print_exc()
    sys.exit(1)
