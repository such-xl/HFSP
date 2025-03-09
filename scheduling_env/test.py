import matplotlib.pyplot as plt

data = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
]

plt.boxplot(data)
plt.xticks([1, 2, 3, 4], ["Group 1", "Group 2", "Group 3", "Group 4"])  # 设置x轴标签
plt.ylabel("Values")  # 设置y轴标签
plt.title("Box Plot of Multiple Groups")  # 设置图表标题
plt.savefig("boxplot.png")
