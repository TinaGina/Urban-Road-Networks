import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置绘图样式
# sns.set(style="whitegrid")

# 读取数据
df = pd.read_excel('GNNTrainTest0120/GNNsWL.xlsx', sheet_name='Sheet2', index_col=0)
print(f'df is: {df}')


features = ['GCN', 'GAT', 'GIN', 'GraphSAGE', 'FiLM', 'EdgeCNN', 'WL']

plt.figure(figsize=(5, 4))

for feature in features:
    sns.lineplot(x=range(1, 21), y=feature, data=df, label=feature)

# plt.title('Standardized Values with Standard Deviation')
plt.xticks()  # rotation=45, ha='right'
plt.xlabel('Epoch')
plt.ylabel('Test accuracy')
# plt.axis('off')
plt.legend(loc='lower left')  # ncol=2
plt.tight_layout()
plt.savefig('GNNTrainTest0120/lineplot')
plt.show()
