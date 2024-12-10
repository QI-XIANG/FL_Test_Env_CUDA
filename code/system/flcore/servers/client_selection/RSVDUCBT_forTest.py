import numpy as np
from sklearn.utils.extmath import randomized_svd
import torch
import torch.distributions as tdist
import math

class RSVDUCBThompsonEnhanced:
    def __init__(self, num_clients, num_join_clients, global_accuracy_history, min_valid_clients=10, c=1, prior_alpha=1, prior_beta=1):
        """
        改進的 RSVD + UCB + Thompson Sampling 演算法
        Enhanced RSVD + UCB + Thompson Sampling algorithm
        """
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        self.min_valid_clients = min_valid_clients
        self.selection_counts = np.zeros(num_clients)  # 客戶端被選擇次數
        self.performance_history = np.zeros(num_clients)  # 客戶端歷史績效
        self.anomaly_score_history = np.zeros(num_clients)  # 異常分數歷史
        self.poisoned_penalty = np.zeros(num_clients)  # 懲罰分數
        self.poisoned_threshold = 0.5  # 初始異常閾值
        self.c = c  # UCB 調節參數
        self.posterior_alpha = torch.ones(num_clients) * prior_alpha  # Thompson Sampling 的 alpha
        self.posterior_beta = torch.ones(num_clients) * prior_beta  # Thompson Sampling 的 beta
        self.decay_factor = 0.9  # 異常分數衰減因子
        self.global_accuracy_history = global_accuracy_history  # 全局模型精度歷史
        self.gradient_stats = {}  # 用於保存梯度統計（均值、方差）
        self.long_term_anomaly_penalty = np.zeros(num_clients)  # 長期異常懲罰

    def calculate_accuracy_change(self):
        """
        計算全局精度的變化量
        Calculate the global accuracy change
        """
        if len(self.global_accuracy_history) < 2:
            return 0.0
        
        recent_changes = np.diff(self.global_accuracy_history[-5:])  # 最近5次精度變化
        
        print("Recent accuracy changes - ", recent_changes)
        
        return np.mean(recent_changes)

    def dynamic_threshold_adjustment(self):
        """
        動態調整異常閾值
        Dynamically adjust the anomaly threshold
        """
        base_threshold = 0.5
        accuracy_trend = self.calculate_accuracy_change()
        if self.global_accuracy_history[-1] > 0.9 or accuracy_trend > 0.01:
            base_threshold = 0.3  # 更嚴格的閾值
        elif self.global_accuracy_history[-1] < 0.8 or accuracy_trend < -0.01:
            base_threshold = 0.6  # 更寬鬆的閾值
        return base_threshold

    def detect_poisoned_clients(self, gradients):
        """
        使用 RSVD 檢測中毒客戶端並更新異常分數
        Detect poisoned clients using RSVD and update anomaly scores
        """
        client_ids = list(gradients.keys())
        gradients_matrix = np.vstack(list(gradients.values()))

        # RSVD 重建
        u, s, vt = randomized_svd(gradients_matrix, n_components=min(gradients_matrix.shape[1], 10), random_state=42)
        reconstructed = np.dot(u, np.dot(np.diag(s), vt))

        # 重建誤差與角度偏差
        reconstruction_errors = np.linalg.norm(gradients_matrix - reconstructed, axis=1)
        gradient_angles = np.arccos(np.clip(
            np.dot(gradients_matrix, vt[0]) / (np.linalg.norm(gradients_matrix, axis=1) * np.linalg.norm(vt[0])), -1, 1))

        # 異常分數（誤差和角度加權）
        normalized_errors = (reconstruction_errors - np.min(reconstruction_errors)) / (np.ptp(reconstruction_errors) + 1e-8)
        normalized_angles = (gradient_angles - np.min(gradient_angles)) / (np.ptp(gradient_angles) + 1e-8)
        combined_scores = 0.7 * normalized_errors + 0.3 * normalized_angles

        # 長期異常分數累加
        for i, client_id in enumerate(client_ids):
            self.anomaly_score_history[client_id] = (
                self.decay_factor * self.anomaly_score_history[client_id] + 
                (1 - self.decay_factor) * combined_scores[i]
            )
            self.long_term_anomaly_penalty[client_id] += combined_scores[i]

        return self.anomaly_score_history

    def select_clients(self, epoch, gradients):
        """
        選擇參與的客戶端
        Select participating clients
        """
        anomaly_scores = self.detect_poisoned_clients(gradients)
        threshold = self.dynamic_threshold_adjustment()
        valid_clients = np.where(anomaly_scores + self.long_term_anomaly_penalty <= threshold, 1, 0)

        # 動態調整 UCB 和 Thompson Sampling 的權重
        weight_ucb = 0.5 + 0.1 * (1 - np.mean(anomaly_scores))  # 異常分數越低，UCB 權重越高
        weight_ts = 1 - weight_ucb

        combined_scores = []
        for i in range(self.num_clients):
            if valid_clients[i] == 1:
                avg_reward = (self.performance_history[i] / self.selection_counts[i]) if self.selection_counts[i] > 0 else 1e-4
                delta_i = math.sqrt(2 * math.log(epoch + 1) / (self.selection_counts[i] + 1))
                ucb_score = avg_reward + self.c * delta_i
                thompson_score = tdist.Beta(self.posterior_alpha[i], self.posterior_beta[i]).sample().item()
                combined_scores.append(weight_ucb * ucb_score + weight_ts * thompson_score)
            else:
                combined_scores.append(-1e400)

        selected_clients = np.argsort(combined_scores)[-self.num_join_clients:]

        # 更新選擇次數
        for client in selected_clients:
            self.selection_counts[client] += 1

        print(f"Epoch {epoch}: Anomaly Scores - {anomaly_scores}")

        return selected_clients

    def update(self, selected_clients, rewards):
        """
        更新績效數據
        Update performance data 
        """
        for client, reward in zip(selected_clients, rewards):
            self.performance_history[client] += reward
            self.posterior_alpha[client] += reward
            self.posterior_beta[client] += (1 - reward)
