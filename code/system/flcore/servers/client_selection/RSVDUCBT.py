import numpy as np
from sklearn.utils.extmath import randomized_svd
import torch
import torch.distributions as tdist
import math

class RSVDUCBThompson:
    def __init__(self, num_clients, num_join_clients, min_valid_clients=10, c=1, prior_alpha=1, prior_beta=1, epsilon=0.1):
        """
        初始化參數，並結合UCB和Thompson Sampling的優勢，適應非IID情況
        :param num_clients: 客戶端總數
        :param num_join_clients: 每輪選擇的客戶端數量
        :param min_valid_clients: 每輪至少保留的有效客戶端數量
        :param c: UCB中的置信區間調節參數
        :param prior_alpha, prior_beta: Thompson Sampling的先驗參數
        :param epsilon: Greedy epsilon機制探索的比例
        """
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        self.min_valid_clients = min_valid_clients
        self.stability_scores = np.ones(num_clients)  # 穩定性分數
        self.trust_scores = np.ones(num_clients)  # 信任分數
        self.selection_counts = np.zeros(num_clients)  # 選擇次數記錄
        self.performance_history = np.zeros(num_clients)  # 歷史績效記錄
        self.anomaly_score_history = np.zeros(num_clients)  # 異常分數歷史
        self.poisoned_threshold = 0.5  # 初始異常閾值
        
        # UCB 和 Thompson Sampling 初始化
        self.c = c
        self.posterior_alpha = torch.ones(num_clients) * prior_alpha
        self.posterior_beta = torch.ones(num_clients) * prior_beta
        self.epsilon = epsilon
        self.client_data_variation = np.random.rand(num_clients)  # 模擬每個客戶端的資料變異度

    def detect_poisoned_clients(self, gradients):
        """
        使用隨機奇異值分解檢測中毒客戶端
        :param gradients: 各客戶端的本地梯度更新 (形狀為 [num_clients, gradient_dim])
        :return: 各客戶端的異常分數
        """
        client_ids = list(gradients.keys())
        gradients = np.vstack(list(gradients.values())) 
        u, s, vt = randomized_svd(gradients, n_components=2, random_state=42)
        reconstructed = np.dot(np.dot(u, np.diag(s)), vt)
        errors = np.linalg.norm(gradients - reconstructed, axis=1)

        # 更新異常分數歷史（平滑處理）
        for i, client_id in enumerate(client_ids):
            self.anomaly_score_history[client_id] = 0.8 * self.anomaly_score_history[client_id] + 0.2 * errors[i]
        
        return self.anomaly_score_history

    def adjust_poisoned_threshold(self, anomaly_scores):
        """
        動態調整 poisoned_threshold
        :param anomaly_scores: 異常分數列表
        :return: 調整後的 poisoned_threshold
        """
        # 計算異常分數的 80% 分位數作為新的閾值
        threshold = np.percentile(anomaly_scores, 80)
        
        # 如果有效客戶端少於最低數量，放寬閾值
        valid_clients = np.where(anomaly_scores <= threshold, 1, 0)
        if np.sum(valid_clients) < self.min_valid_clients:
            # 放寬閾值至 90% 分位數
            threshold = np.percentile(anomaly_scores, 90)
            print(f"Relaxed anomaly threshold to {threshold}")
        
        self.poisoned_threshold = threshold

    def select_clients(self, epoch, gradients):
        """
        客戶端選擇邏輯
        :param epoch: 當前訓練輪次
        :param gradients: 各客戶端的本地梯度更新
        :return: 選定的客戶端列表
        """
        anomaly_scores = self.detect_poisoned_clients(gradients)

        # 動態調整 poisoned_threshold
        self.adjust_poisoned_threshold(anomaly_scores)

        # 根據 adjusted threshold 選擇有效客戶端
        valid_clients = np.where(anomaly_scores <= self.poisoned_threshold, 1, 0)
        
        # 計算 UCB 和 Thompson Sampling 分數，並考慮資料變異度
        ucb_scores = []
        thompson_scores = []
        for i in range(self.num_clients):
            if valid_clients[i] == 1:
                # UCB 計算：資料變異度越大，UCB的置信區間越大
                data_variation_factor = self.client_data_variation[i] + 0.5  # 加入資料變異度因子
                if self.selection_counts[i] > 0:
                    avg_reward = self.performance_history[i] / self.selection_counts[i]
                    delta_i = math.sqrt(2 * math.log(epoch + 1) / (self.selection_counts[i] * data_variation_factor))
                    ucb_score = avg_reward + self.c * delta_i
                else:
                    ucb_score = np.random.uniform(1.5, 2.5)

                # Thompson Sampling 計算：結合資料變異度進行調整
                thompson_sample = tdist.Beta(self.posterior_alpha[i], self.posterior_beta[i]).sample().item()
                thompson_score = thompson_sample * (1 + self.client_data_variation[i] * 0.2)  # 資料變異度影響

                ucb_scores.append(ucb_score)
                thompson_scores.append(thompson_score)
            else:
                ucb_scores.append(-1e400)
                thompson_scores.append(-1e400)

        # 結合 UCB 和 Thompson Sampling 分數
        combined_scores = np.array(ucb_scores) * 0.5 + np.array(thompson_scores) * 0.5
        
        # 隨機性噪聲
        noise_level = max(0.05, np.std(combined_scores) * 0.1)
        noise = np.random.uniform(0, noise_level, self.num_clients)
        adjusted_scores = combined_scores + noise

        # 隨機探索 (Greedy epsilon機制)
        if np.random.rand() < self.epsilon:
            print(f"Epoch {epoch}: Exploration activated.")
            selected_clients = np.random.choice(np.where(valid_clients == 1)[0], self.num_join_clients, replace=False)
        else:
            # 選擇分數最高的 num_join_clients 客戶端
            selected_clients = np.argsort(adjusted_scores)[-self.num_join_clients:]

        # 更新選擇次數
        for client in selected_clients:
            self.selection_counts[client] += 1

        print(f"Epoch {epoch}: Anomaly Scores - {anomaly_scores}")
        return selected_clients

    def update(self, selected_clients, rewards):
        """
        更新信任分數與穩定性分數
        :param selected_clients: 被選中的客戶端列表
        :param rewards: 客戶端的績效分數
        """
        # 計算當前選定客戶端的平均績效作為基準
        avg_rewards = np.mean(rewards)

        for client, reward in zip(selected_clients, rewards):
            # 使用移動平均更新歷史績效
            self.performance_history[client] = (
                self.performance_history[client] * 0.8 + reward * 0.2
            )

            # 動態調整更新幅度（根據與基準的差距）
            delta = abs(reward - avg_rewards) / (avg_rewards + 1e-8)  # 防止除零
            if reward < 0:  # 表現不佳，降低分數
                self.trust_scores[client] *= max(1 - delta, 0.8)
                self.stability_scores[client] *= max(1 - delta, 0.8)
            else:  # 表現良好，提高分數
                self.trust_scores[client] = min(self.trust_scores[client] + delta * 0.1, 1.0)
                self.stability_scores[client] = min(self.stability_scores[client] + delta * 0.1, 1.0)
