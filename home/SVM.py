import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

"""训练SVM分类器，对比exp2中保存的2中攻击改写方法文案对训练的分类器的攻击成功率"""

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体为 Microsoft YaHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 预处理
def clean_text(text):
    """文本清洗函数"""
    if not isinstance(text, str):
        return ""
    text = text.replace("音频内容：", "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text



# 训练分类器
print("Step 1: 正在加载并预处理训练数据...")
try:
    train_df = pd.read_csv('训练集结果.csv')
    train_df = train_df.dropna(subset=['is_fraud'])  # 去除标签为空的行
    train_df['clean_text'] = train_df['specific_dialogue_content'].apply(clean_text)
    train_df['label'] = train_df['is_fraud'].astype(int)

    print("Step 2: 正在训练 SVM 分类器...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('svm', SVC(kernel='linear', probability=True, random_state=42))
    ])
    pipeline.fit(train_df['clean_text'], train_df['label'])

    # 保存模型
    joblib.dump(pipeline, 'svm_fraud_classifier.pkl')
    print("        SVM 模型训练完成并已保存。")

except Exception as e:
    print(f"训练过程中出错: {e}")
    exit()


# 加载实验数据并评估
print("Step 3: 正在加载 Exp2 实验数据...")
try:
    exp_df = pd.read_csv('exp2_paper_method_results.csv')
except FileNotFoundError:
    print("错误：未找到 'exp2_paper_method_results.csv'。")
    exit()

# 全为 1 (诈骗)
ground_truth = [1] * len(exp_df)

# 计算 SVM 的 ASR ---
# Method A
exp_df['clean_method_a'] = exp_df['method_a_text'].apply(clean_text)
pred_svm_a = pipeline.predict(exp_df['clean_method_a'])
acc_svm_a = accuracy_score(ground_truth, pred_svm_a)
asr_svm_a = 1 - acc_svm_a  # 攻击成功率 = 1 - 检测准确率

# Method B
exp_df['clean_method_b'] = exp_df['method_b_text'].apply(clean_text)
pred_svm_b = pipeline.predict(exp_df['clean_method_b'])
acc_svm_b = accuracy_score(ground_truth, pred_svm_b)
asr_svm_b = 1 - acc_svm_b

# 获取大模型的 ASR ---
asr_qianwen_a = exp_df['method_a_success'].astype(int).mean()
asr_qianwen_b = exp_df['method_b_success'].astype(int).mean()


# 可视化
plot_data = pd.DataFrame({
    'Method': ['Method A\n(Word Substitution)', 'Method A\n(Word Substitution)',
               'Method B\n(Style Rewriting)', 'Method B\n(Style Rewriting)'],
    'Model': ['SVM', 'Qianwen', 'SVM', 'Qianwen'],
    'ASR': [asr_svm_a, asr_qianwen_a, asr_svm_b, asr_qianwen_b]
})

print("\n" + "=" * 40)
print("        模型鲁棒性对比结果 (ASR)        ")
print("=" * 40)
print(plot_data)

# 绘制柱状图
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

palette = {"SVM": "#4c72b0", "Qianwen": "#c44e52"}
ax = sns.barplot(x='Method', y='ASR', hue='Model', data=plot_data, palette=palette)

# 在柱状图上方标注具体数值
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.text(p.get_x() + p.get_width() / 2., height + 0.005,
                f'{height:.1%}', ha="center", va="bottom", fontsize=12, fontweight='bold')

plt.ylim(0, max(plot_data['ASR']) * 1.3)  # 留出顶部空间
plt.title('Attack Success Rate (ASR) Comparison', fontsize=16, pad=20)
plt.ylabel('Attack Success Rate', fontsize=14)
plt.xlabel('Rewriting Method', fontsize=14)
plt.legend(title='Target Model', fontsize=12, loc='upper left')

plt.tight_layout()

# 保存图片
plot_filename = 'asr_comparison_svm_vs_qianwen.png'
plt.savefig(plot_filename, dpi=300)
print(f"\n对比图表已保存至: {plot_filename}")

# 保存详细数据
exp_df['svm_pred_method_a'] = pred_svm_a
exp_df['svm_pred_method_b'] = pred_svm_b
output_csv = 'exp2_svm_classification_results.csv'
exp_df.to_csv(output_csv, index=False)
print(f"详细分类结果已保存至: {output_csv}")