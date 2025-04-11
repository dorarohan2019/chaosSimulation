import gym
from gym import spaces
import numpy as np
import boto3
from collections import deque
import torch
from predictive_model_grok import LSTMAutoencoder
from stable_baselines3 import DQN
import random

class RemediationEnvironment(gym.Env):
    def __init__(self, is_local=True, use_collected_states=True, load_model=False):
        super(RemediationEnvironment, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(37)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5, 25), dtype=np.float32)

        # LocalStack endpoint
        self.endpoint = 'http://localhost:4566' if is_local else None
        self.session = boto3.Session(
            aws_access_key_id='dummy',
            aws_secret_access_key='dummy',
            region_name='us-east-1'
        )

        # Initialize AWS service clients
        self.ec2 = self.session.client('ec2', endpoint_url=self.endpoint, region_name='us-east-1')
        self.s3 = self.session.client('s3', endpoint_url=self.endpoint)
        self.rds = self.session.client('rds', endpoint_url=self.endpoint, region_name='us-east-1')
        self.lambda_ = self.session.client('lambda', endpoint_url=self.endpoint, region_name='us-east-1')
        self.elb = self.session.client('elb', endpoint_url=self.endpoint, region_name='us-east-1')
        self.dynamodb = self.session.client('dynamodb', endpoint_url=self.endpoint, region_name='us-east-1')
        self.iam = self.session.client('iam', endpoint_url=self.endpoint)
        self.cloudtrail = self.session.client('cloudtrail', endpoint_url=self.endpoint, region_name='us-east-1')
        self.waf = self.session.client('wafv2', endpoint_url=self.endpoint, region_name='us-east-1')
        self.apigateway = self.session.client('apigateway', endpoint_url=self.endpoint, region_name='us-east-1')

        # State boundaries
        self.state_min = np.array([5, 0, 0, 2, 0, 0, 0, 1, 100, 0.01, 3, 50, 0, 100, 10, 1000, 1e9, 2, 0, 0, 0, 0, 1e5, 1e5, 0], dtype=np.float32)
        self.state_max = np.array([5, 5, 100, 2, 2, 200, 100, 1, 2000, 0.5, 3, 500, 10, 1000, 10, 5000, 5e9, 2, 500, 20, 50, 10, 5e6, 5e6, 10], dtype=np.float32)

        # Define optimal values for key metrics
        self.optimal_values = {
            1: 5.0,      # ec2_running
            2: 50.0,     # ec2_cpu_avg
            4: 2.0,      # rds_available
            5: 100.0,    # rds_connections
            6: 50.0,     # rds_cpu
            8: 1000.0,   # elb_requests
            9: 0.01,     # elb_latency
            11: 250.0,   # lambda_invocations
            12: 0.0,     # lambda_errors
            13: 100.0,   # lambda_duration
            15: 3000.0,  # s3_object_count
            16: 3e9,     # s3_total_size
            18: 0.0,     # sqs_message_count
            19: 0.0,     # security_findings
            20: 0.0,     # failed_logins
            21: 0.0,     # vulnerability_count
            22: 3e6,     # network_in
            23: 3e6,     # network_out
            24: 0.0      # packet_loss_percent
        }

        # Load DQN model
        if load_model:
            try:
                self.model = DQN.load("remediation_agent")
                print("DQN model loaded successfully.")
            except FileNotFoundError:
                print("DQN model not found. Please train the model first.")
                self.model = None
        else:
            self.model = None

        # Load predictive model
        self.predictive_model = LSTMAutoencoder(timesteps=5, features=25)
        try:
            self.predictive_model.load_state_dict(torch.load('lstm_autoencoder.pth'))
            self.predictive_model.eval()
            print("LSTMAutoencoder model loaded successfully.")
        except FileNotFoundError:
            print("LSTMAutoencoder model not found. Please train the model first.")
            self.predictive_model = None

        # State history and action costs
        self.state_history = deque(maxlen=5)
        self.action_costs = {i: 0.5 for i in range(37)}

        # Remediation actions
        self.remediation_actions = {
            0: self.restore_ec2_instance,
            1: self.start_ec2_instance,
            2: self.relieve_cpu_stress,
            3: self.relieve_memory_stress,
            4: self.relieve_disk_stress,
            5: self.remove_network_delay,
            6: self.remove_packet_loss,
            7: self.restore_s3_object,
            8: self.fix_s3_permissions,
            9: self.remove_s3_throttling,
            10: self.enable_lambda_function,
            11: self.reset_lambda_timeout,
            12: self.relieve_lambda_memory_pressure,
            13: self.remove_lambda_concurrency_limit,
            14: self.restore_dynamodb_throughput,
            15: self.restore_dynamodb_items,
            16: self.enable_dynamodb_table,
            17: self.restore_iam_permissions,
            18: self.remove_restrictive_policy,
            19: self.reset_access_keys,
            20: self.unblock_api_requests,
            21: self.fix_dns_failure,
            22: self.restore_connection_draining,
            23: self.remove_latency_injection,
            24: self.restore_rds_failover,
            25: self.relieve_rds_storage_pressure,
            26: self.relieve_rds_connection_flood,
            27: self.relieve_rds_cpu_stress,
            28: self.register_elb_instance,
            29: self.restore_elb_availability_zone,
            30: self.mitigate_security_breach,
            31: self.block_brute_force_login,
            32: self.mitigate_ddos,
            33: self.prevent_data_exfiltration,
            34: self.enable_cloudwatch_alarms,
            35: self.restore_cloudtrail_logs,
            36: self.no_action
        }

        # Enhanced remediation effects for all 37 actions
        self.remediation_action_effects = {
            0: {1: 1, 2: -10},            # restore_ec2_instance
            1: {1: 1, 2: -5},             # start_ec2_instance
            2: {2: -20, 6: -10},          # relieve_cpu_stress
            3: {2: -15, 6: -5},           # relieve_memory_stress
            4: {15: -100, 16: -1e8},      # relieve_disk_stress
            5: {9: -0.05, 24: -2},        # remove_network_delay
            6: {24: -3, 9: -0.02},        # remove_packet_loss
            7: {15: -50, 16: -5e7},       # restore_s3_object
            8: {19: -5, 15: -20},         # fix_s3_permissions
            9: {8: 100, 9: -0.03},        # remove_s3_throttling
            10: {11: -20, 12: -2},        # enable_lambda_function
            11: {13: -50, 11: -10},       # reset_lambda_timeout
            12: {11: -15, 12: -1},        # relieve_lambda_memory_pressure
            13: {11: -25, 13: -20},       # remove_lambda_concurrency_limit
            14: {22: 5e5, 23: 5e5},       # restore_dynamodb_throughput
            15: {15: -100, 16: -1e8},     # restore_dynamodb_items
            16: {22: 1e5, 23: 1e5},       # enable_dynamodb_table
            17: {19: -10, 20: -5},        # restore_iam_permissions
            18: {19: -8, 21: -2},         # remove_restrictive_policy
            19: {20: -10, 19: -3},        # reset_access_keys
            20: {8: 200, 9: -0.04},       # unblock_api_requests
            21: {24: -1, 9: -0.01},       # fix_dns_failure
            22: {8: 150, 2: -5},          # restore_connection_draining
            23: {9: -0.1, 24: -1},        # remove_latency_injection
            24: {4: 1, 5: -20},           # restore_rds_failover
            25: {6: -10, 5: -15},         # relieve_rds_storage_pressure
            26: {5: -25, 6: -5},          # relieve_rds_connection_flood
            27: {6: -20, 2: -5},          # relieve_rds_cpu_stress
            28: {8: 300, 9: -0.02},       # register_elb_instance
            29: {8: 250, 24: -1},         # restore_elb_availability_zone
            30: {19: -15, 20: -10},       # mitigate_security_breach
            31: {20: -20, 21: -3},        # block_brute_force_login
            32: {24: -5, 9: -0.05},       # mitigate_ddos
            33: {19: -10, 21: -5},        # prevent_data_exfiltration
            34: {16: -5e7, 15: -50},      # enable_cloudwatch_alarms
            35: {16: -1e8, 19: -2},       # restore_cloudtrail_logs
            36: {}                         # no_action
        }

        # Load collected states
        self.use_collected_states = use_collected_states
        if self.use_collected_states:
            try:
                self.collected_states = np.load("environment_states_grok.npy")
                self.labels = np.load("labels.npy", allow_pickle=True)
            except FileNotFoundError:
                print("Collected states or labels file not found. Falling back to default state.")
                self.collected_states = None
                self.labels = None
        else:
            self.collected_states = None
            self.labels = None

        self.reset()

    # Remediation action placeholders
    def restore_ec2_instance(self): pass
    def start_ec2_instance(self): pass
    def relieve_cpu_stress(self): pass
    def relieve_memory_stress(self): pass
    def relieve_disk_stress(self): pass
    def remove_network_delay(self): pass
    def remove_packet_loss(self): pass
    def restore_s3_object(self): pass
    def fix_s3_permissions(self): pass
    def remove_s3_throttling(self): pass
    def enable_lambda_function(self): pass
    def reset_lambda_timeout(self): pass
    def relieve_lambda_memory_pressure(self): pass
    def remove_lambda_concurrency_limit(self): pass
    def restore_dynamodb_throughput(self): pass
    def restore_dynamodb_items(self): pass
    def enable_dynamodb_table(self): pass
    def restore_iam_permissions(self): pass
    def remove_restrictive_policy(self): pass
    def reset_access_keys(self): pass
    def unblock_api_requests(self): pass
    def fix_dns_failure(self): pass
    def restore_connection_draining(self): pass
    def remove_latency_injection(self): pass
    def restore_rds_failover(self): pass
    def relieve_rds_storage_pressure(self): pass
    def relieve_rds_connection_flood(self): pass
    def relieve_rds_cpu_stress(self): pass
    def register_elb_instance(self): pass
    def restore_elb_availability_zone(self): pass
    def mitigate_security_breach(self): pass
    def block_brute_force_login(self): pass
    def mitigate_ddos(self): pass
    def prevent_data_exfiltration(self): pass
    def enable_cloudwatch_alarms(self): pass
    def restore_cloudtrail_logs(self): pass
    def no_action(self): pass

    # Environment methods
    def normalize_state(self, state):
        normalized = (state - self.state_min) / (self.state_max - self.state_min + 1e-8)
        return np.clip(normalized, 0, 1)

    def denormalize_state(self, normalized_state):
        return normalized_state * (self.state_max - self.state_min) + self.state_min

    def reset(self):
        if self.use_collected_states and self.collected_states is not None and len(self.collected_states) >= 5:
            if self.labels is not None and len(self.labels) >= len(self.collected_states):
                anomalous_indices = [i for i, label in enumerate(self.labels) if label != 'normal' and i <= len(self.collected_states) - 5]
                if anomalous_indices:
                    start_idx = random.choice(anomalous_indices)
                else:
                    start_idx = random.randint(0, len(self.collected_states) - 5)
            else:
                start_idx = random.randint(0, len(self.collected_states) - 5)
            self.state_history.clear()
            for i in range(5):
                state = self.collected_states[start_idx + i].astype(np.float32)
                normalized_state = self.normalize_state(state)
                self.state_history.append(normalized_state.copy())
            self.state = self.state_history[-1].copy()
        else:
            raw_state = np.array([5, 3, 50.0, 2, 1, 100, 50.0, 1, 1000, 0.25, 3, 250, 5, 500, 10, 3000, 3e9, 2, 250, 10, 25, 5, 3e6, 3e6, 5], dtype=np.float32)
            self.state = self.normalize_state(raw_state)
            self.state_history.clear()
            for _ in range(5):
                self.state_history.append(self.state.copy())
        return np.array(self.state_history)

    def step(self, action):
        state_sequence_before = np.array(self.state_history)
        anomaly_score_before = self.predictive_model.get_anomaly_score(state_sequence_before) if self.predictive_model else 0.5
        raw_state_before = self.denormalize_state(self.state)
        raw_state = raw_state_before.copy()

        try:
            self.remediation_actions[action]()
            #print(f"Performing action: {action}\n")

            # Apply state changes with dynamic deltas for metrics in optimal_values
            for idx, delta in self.remediation_action_effects.get(action, {}).items():
                if idx in self.optimal_values:
                    current_value = raw_state[idx]
                    optimal_value = self.optimal_values[idx]
                    if current_value > optimal_value:
                        delta = min(-10, -0.1 * (current_value - optimal_value))
                    else:
                        delta = max(10, -0.1 * (current_value - optimal_value))
                raw_state[idx] = max(self.state_min[idx], min(self.state_max[idx], raw_state[idx] + delta))
                #print(f"Applying remediation action {action} for index: {idx} with delta: {delta}, new value: {raw_state[idx]}\n")

            self.state = self.normalize_state(raw_state)
            self.state_history.append(self.state.copy())
            state_sequence_after = np.array(self.state_history)
            anomaly_score_after = self.predictive_model.get_anomaly_score(state_sequence_after) if self.predictive_model else 0.5

            # Calculate reward components
            threat_reduction = max(0, anomaly_score_before - anomaly_score_after)
            business_impact = abs(raw_state[1] - 5)  # Target all EC2 instances running
            action_cost = self.action_costs[action]
            key_metrics = list(self.optimal_values.keys())
            distance_reduced = sum(
                max(0, abs(raw_state_before[i] - self.optimal_values[i]) - abs(raw_state[i] - self.optimal_values[i]))
                for i in key_metrics
            )
            reward = (1.0 * threat_reduction) - (0.01 * action_cost) - (0.05 * business_impact) + (0.01 * distance_reduced)

            done = (anomaly_score_after < 0.1) or (raw_state[1] <= 0) or (len(self.state_history) > 100)
            with open("remediation_log.txt", "a") as f:
                    f.write(f"Action {action}, State {self.state.tolist()}, Reward {reward}, "
                    f"Anomaly score: {anomaly_score_before} -> {anomaly_score_after}, "
                    f"Threat Reduction: {threat_reduction}, Business Impact: {business_impact}, "
                    f"Action Cost: {action_cost}\n")
            #print(f"Action {action}"#, State {self.state.tolist()}, Reward {reward}, "
            #      f"Anomaly score: {anomaly_score_before} -> {anomaly_score_after}, "
            #      f"Threat Reduction: {threat_reduction}, Business Impact: {business_impact}, "
            #      f"Action Cost: {action_cost}, Distance Reduced: {distance_reduced}\n")
            #print(f"Action {action}, Before: {raw_state_before.tolist()}, After: {raw_state.tolist()}\n")

            return state_sequence_after, reward, done, {}

        except Exception as e:
            print(f"Remediation action {action} failed: {e}")
            return state_sequence_before, 0, False, {}

    def select_action(self, state):
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        state = np.array(state)
        if state.shape == (1, 5, 25):
            state = state.squeeze(0)
        elif state.shape != (5, 25):
            raise ValueError(f"Invalid state shape: {state.shape}, expected (5, 25)")
        anomaly_score = self.predictive_model.get_anomaly_score(state) if self.predictive_model else 0.5
        if self.model is None:
            print("DQN model not loaded. Selecting random action.")
            return np.random.randint(0, 37)
        if anomaly_score > 0.1:
            action, _ = self.model.predict(state, deterministic=True)
            print(f"Action: {action} selected")
            return action
        action = np.random.randint(0, 37)
        print(f"Random action: {action} selected as anomaly_score is < 0.1")
        return action

def train_remediation_agent(is_local=True, use_collected_states=True):
    env = RemediationEnvironment(is_local=is_local, use_collected_states=use_collected_states)
    model = DQN(
        "MlpPolicy",
        env,
        verbose=2,
        learning_rate=0.001,
        exploration_fraction=0.5,
        buffer_size=10000,
        batch_size=32
    )
    model.learn(total_timesteps=100000)
    model.save("remediation_agent")
    return model

if __name__ == "__main__":
    train_remediation_agent(is_local=True, use_collected_states=True)