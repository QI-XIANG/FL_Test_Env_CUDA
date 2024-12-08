import numpy as np
import math
from sklearn.utils.extmath import randomized_svd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
import copy

class RSVDUCBClientSelection:
    def __init__(self, num_clients, num_join_clients, min_valid_clients=10, c=1):
        """
        初始化參數
        :param num_clients: 客戶端總數
        :param num_join_clients: 每輪選擇的客戶端數量
        :param min_valid_clients: 每輪至少保留的有效客戶端數量
        :param c: UCB 中的置信區間調節參數
        """
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        self.min_valid_clients = min_valid_clients
        self.c = c

        # RSVD 部分
        self.anomaly_scores = np.zeros(num_clients)
        self.poisoned_threshold = 0.5  # 初始閾值

        # UCB 部分
        self.numbers_of_selections = np.zeros(num_clients)
        self.sums_of_rewards = np.zeros(num_clients)

        # 孤立森林模型
        self.isolation_forest = IsolationForest()
        
        # ARIMA 時序模型
        self.arima_models = [None] * num_clients
        self.arima_past_rewards = [list() for _ in range(num_clients)]

    def detect_poisoned_clients(self, gradients):
        """
        使用 RSVD 和孤立森林檢測異常分數
        :param gradients: 各客戶端的本地梯度更新 (形狀為 [num_clients, gradient_dim])
        :return: 各客戶端的異常分數
        """
        client_ids = list(gradients.keys())
        gradients = np.vstack(list(gradients.values()))
        
        # RSVD 部分
        u, s, vt = randomized_svd(gradients, n_components=2, random_state=42)
        reconstructed = np.dot(np.dot(u, np.diag(s)), vt)
        errors = np.linalg.norm(gradients - reconstructed, axis=1)

        # 孤立森林檢測
        isolation_scores = self.isolation_forest.fit_predict(gradients)
        isolation_scores = np.where(isolation_scores == 1, 0, 1)  # 將 -1 (異常) 轉為 1，1 (正常) 轉為 0

        # 平滑異常分數
        for i, client_id in enumerate(client_ids):
            self.anomaly_scores[client_id] = 0.8 * self.anomaly_scores[client_id] + 0.2 * (errors[i] + isolation_scores[i])

        return self.anomaly_scores

    def adjust_threshold(self):
        """
        動態調整異常分數的閾值
        """
        median_score = np.median(self.anomaly_scores)
        self.poisoned_threshold = median_score + 0.1  # 動態偏移
        print(f"Updated poisoned threshold to {self.poisoned_threshold}")

    def get_n_max(self, n, target):
        """
        獲取前 n 大的索引
        :param n: 要選擇的最大數量
        :param target: 目標數組
        :return: 前 n 大數值的索引
        """
        t = copy.deepcopy(target)
        max_index = []
        for _ in range(n):
            index = np.argmax(t)
            t[index] = -1e400  # 排除已選擇的最大值
            max_index.append(index)
        return max_index

    def select_clients(self, epoch, gradients):
        """
        混合式選擇邏輯
        :param epoch: 當前訓練輪次
        :param gradients: 各客戶端的本地梯度更新
        :return: 選定的客戶端列表
        """
        # RSVD + 孤立森林: 檢測異常分數
        anomaly_scores = self.detect_poisoned_clients(gradients)

        # 動態調整異常閾值
        self.adjust_threshold()

        # 初步篩選有效客戶端
        valid_clients = np.where(anomaly_scores <= self.poisoned_threshold, 1, 0)

        # 動態調整閾值，確保最小有效客戶端數量
        if np.sum(valid_clients) < self.min_valid_clients:
            relaxed_threshold = np.percentile(anomaly_scores, 80)
            valid_clients = np.where(anomaly_scores <= relaxed_threshold, 1, 0)
            print(f"Relaxed anomaly threshold to {relaxed_threshold}")

        # UCB: 計算上限置信區間
        clients_upper_bound = np.full(self.num_clients, -1e400)
        for i in range(self.num_clients):
            if valid_clients[i] == 1:
                if self.numbers_of_selections[i] > 0:
                    average_reward = self.sums_of_rewards[i] / self.numbers_of_selections[i]
                    delta_i = math.sqrt(2 * math.log(epoch + 1) / self.numbers_of_selections[i])
                    clients_upper_bound[i] = average_reward + self.c * delta_i
                else:
                    clients_upper_bound[i] = 1e400

        # 選擇分數最高的 num_join_clients 客戶端
        selected_clients = self.get_n_max(self.num_join_clients, clients_upper_bound)

        # 更新選擇次數
        for client in selected_clients:
            self.numbers_of_selections[client] += 1

        #print(f"Epoch {epoch}: Selected Clients - {selected_clients}")
        print(f"Epoch {epoch}: Anomaly Scores - {anomaly_scores}")
        return selected_clients

    def update(self, selected_clients, rewards):
        """
        更新信任分數與穩定性分數
        :param selected_clients: 被選中的客戶端列表
        :param rewards: 客戶端的績效分數
        """
        # 更新每個客戶端的歷史獎勳
        for client, reward in zip(selected_clients, rewards):
            self.sums_of_rewards[client] += reward
            self.numbers_of_selections[client] += 1

            # 記錄客戶端的獎勳進行 ARIMA 時序預測
            self.arima_past_rewards[client].append(reward)

            # 如果歷史資料足夠長，訓練 ARIMA 模型
            if len(self.arima_past_rewards[client]) > 5:
                arima_model = ARIMA(self.arima_past_rewards[client], order=(1, 0, 1))  # ARIMA(1, 0, 1)
                arima_fit = arima_model.fit()
                self.arima_models[client] = arima_fit

            # 如果 ARIMA 預測出異常獎勳，則進行進一步的篩選
            if self.arima_models[client]:
                arima_forecast = self.arima_models[client].forecast(steps=1)
                if abs(reward - arima_forecast[0]) > 0.5:  # 如果偏差過大，視為異常
                    print(f"Client {client} shows abnormal reward behavior, with forecasted reward of {arima_forecast[0]}")
                    self.anomaly_scores[client] = 1.0  # 標記為異常

        #print("Updated Rewards:", self.sums_of_rewards)
        #print("Updated Selections:", self.numbers_of_selections)
