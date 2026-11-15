
import csv,json,sys,numpy as np
sys.path.insert(0,str(__import__("pathlib").Path.cwd()))
from pathlib import Path
from fault_workflow.figures import plt
from fault_workflow.run_review import fixed_score
from fault_workflow.common import write_json,timestamp,sha256
root=Path('outputs/server_balanced_v2')
base=root/'baseline/faultseg_balanced_v2'
author=json.loads((base/'author/synthetic/evaluation_report.json').read_text(encoding='utf-8'))
trained=json.loads((base/'trained/synthetic/evaluation_report.json').read_text(encoding='utf-8'))
history=root/'training/faultseg_balanced_v2/history.csv'
with history.open(encoding='utf-8',newline='') as stream: rows=list(csv.DictReader(stream))
assert len(rows)==100
epochs=[int(r['epoch']) for r in rows]
fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
for name,label in [('train_loss','训练'),('validation_loss','验证（原体及同步翻转）')]:
    axes[0].plot(epochs,[float(r[name]) for r in rows],label=label)
axes[0].axvline(60,color='black',linestyle='--',label='预先指定第60轮')
axes[0].set(xlabel='轮次',ylabel='平衡交叉熵损失',title='完整源域训练；验证不参与梯度更新')
axes[0].legend()
a=[fixed_score({'aggregate':c})['dice'] for c in author['cases']]
b=[fixed_score({'aggregate':c})['dice'] for c in trained['cases']]
axes[1].plot(range(20),a,'o-',label='作者模型')
axes[1].plot(range(20),b,'o-',label='复现第60轮')
axes[1].set(xlabel='原始验证体编号',ylabel='Dice',title='20个完整合成验证体；阈值固定0.5',xticks=list(range(0,20,2)))
axes[1].legend()
fig.savefig(root/'review/source_validation_comparison.png',dpi=160);plt.close(fig)
result={'status':'passed_source_gate_only','finished':timestamp(),'author':fixed_score(author),'trained':fixed_score(trained),
        'dice_delta':fixed_score(trained)['dice']-fixed_score(author)['dice'],
        'minimum_per_volume_dice_delta':float(np.min(np.asarray(b)-a)),'all_twenty_dice_improved':bool(np.all(np.asarray(b)>a)),
        'late_validation_note':'Validation loss does not decrease monotonically with training loss; no evidence for blindly adding epochs. Fixed epoch60 selection remains unchanged.',
        'thebe_quality_accepted':False,'test_used':False,'history_sha256':sha256(history)}
write_json(root/'review/source_stage_result.json',result)
