from matplotlib import pyplot as plt
import numpy as np

# ax1 = plt.subplot()
# t = np.arange(0.01, 10.0, 0.01)
# s1 = np.exp(t)
# plt.plot(t,s1,'b-')
# plt.xlabel('t (s)')
# plt.ylabel('exp',color='b')

# ax2 = ax1.twinx()
# s2 = np.sin(2*np.pi*t)
# ax2.plot(t, s2, 'r.')
# plt.ylabel('sin', color='r')
# plt.show()

folds = [str(fold) for fold in range(1, 6)]
mae = [100, 200, 300, 400, 500]
mse = [10, 20, 30, 40, 50]
ax1 = plt.subplot()
ax1.bar(np.arange(len(folds)) - 0.21, mae, 0.4, label='MAE', color='maroon')
plt.xticks(np.arange(len(folds)), folds)
plt.xlabel("Folds")
plt.ylabel("MAE")

ax2 = ax1.twinx()
ax2.bar(np.arange(len(folds)) + 0.21, mse, 0.4, label='MSE', color='green')
plt.ylabel('MSE')
plt.title("EVALUATION METRIC")
plt.legend()
plt.savefig('chart.png')
plt.show()