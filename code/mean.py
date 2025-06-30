import json
import numpy as np

with open('./multimodal_bart_model/bart_fed_eval_results.json', 'r') as f:
    data = json.load(f)

rouge = data.get('rouge', {})
bert = data.get('bertscore', {})

bert_means = {
    'precision': float(np.mean(bert.get('precision', []))),
    'recall':    float(np.mean(bert.get('recall', []))),
    'f1':        float(np.mean(bert.get('f1', [])))
}

summary_means = {
    'rouge': rouge,
    'bertscore': bert_means
}

output_path = './multimodal_bart_model/bart_fed_eval_results_mean.json'
with open(output_path, 'w') as f:
    json.dump(summary_means, f, indent=4)