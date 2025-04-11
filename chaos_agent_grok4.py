import gym
from gym import spaces
import numpy as np
import boto3
from collections import deque
import torch
from predictive_model_grok import LSTMAutoencoder
from stable_baselines3 import DQN
import os
import time
import random
from botocore.exceptions import ClientError
import zipfile
import io

class ChaosEnvironment(gym.Env):
    def __init__(self, is_local=True, use_collected_states=True, load_model=False):
        super(ChaosEnvironment, self).__init__()
        self.action_space = spaces.Discrete(37)  # 37 chaos actions
        self.observation_space = spaces.Box(low=0, high=1, shape=(5, 25), dtype=np.float32)
        self.endpoint = 'http://localhost:4566' if is_local else None
        self.session = boto3.Session(
            aws_access_key_id='dummy',
            aws_secret_access_key='dummy',
            region_name='us-east-1'
        )
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

        # Define realistic bounds for each feature based on StateCollector metric_keys
        self.state_min = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.state_max = np.array([2, 2, 100, 1, 1, 200, 100, 1, 1000, 5.0, 2, 1000, 20, 2000.0, 2, 10000, 50.0, 1, 1000, 15, 50, 40, 10000, 15000, 20], dtype=np.float32)
        # Explanation of bounds:
        # 0: ec2_count (0-2), 1: ec2_running (0-2), 2: ec2_cpu_avg (0-100%), 3: rds_count (0-1), 4: rds_available (0-1),
        # 5: rds_connections (0-200), 6: rds_cpu (0-100%), 7: elb_count (0-1), 8: elb_requests (0-1000), 9: elb_latency (0-5.0s),
        # 10: lambda_count (0-2), 11: lambda_invocations (0-1000), 12: lambda_errors (0-20), 13: lambda_duration (0-2000ms),
        # 14: s3_bucket_count (0-2), 15: s3_object_count (0-10000), 16: s3_total_size (0-50MB), 17: sqs_queue_count (0-1),
        # 18: sqs_message_count (0-1000), 19: security_findings (0-15), 20: failed_logins (0-50), 21: vulnerability_count (0-40),
        # 22: network_in (0-10000MB), 23: network_out (0-15000MB), 24: packet_loss_percent (0-20%)

        if load_model:
            try:
                self.model = DQN.load("chaos_agent")
                print("DQN model loaded successfully.")
            except FileNotFoundError:
                print("DQN model not found. Please train the model first.")
                self.model = None
        else:
            self.model = None

        self.predictive_model = LSTMAutoencoder(timesteps=5, features=25)
        try:
            self.predictive_model.load_state_dict(torch.load('lstm_autoencoder.pth'))
            self.predictive_model.eval()
            print("LSTMAutoencoder model loaded successfully.")
        except FileNotFoundError:
            print("LSTMAutoencoder model not found. Please train the model first.")

        self.state_history = deque(maxlen=5)
        self.action_costs = {i: 0.5 for i in range(37)}
        response = self.ec2.describe_instances(Filters=[{'Name': 'tag:Name', 'Values': ['ChaosTest']}])
        instances = [inst for res in response.get('Reservations', []) for inst in res.get('Instances', [])]
        self.instance_ids = [inst['InstanceId'] for inst in instances] if instances else ['i-123']
        response = self.ec2.describe_security_groups(Filters=[{'Name': 'group-name', 'Values': ['chaos-sg']}])
        self.sg_id = response['SecurityGroups'][0]['GroupId'] if response['SecurityGroups'] else 'sg-123'
        response = self.apigateway.get_rest_apis()
        self.api_id = response['items'][0]['id'] if response['items'] else 'api-123'

        self.chaos_action_effects = {
            0: {0: -1, 10: 5}, 1: {14: 20}, 2: {2: -1}, 3: {3: -1}, 4: {8: 100}, 5: {5: -2}, 6: {6: 0.5, 12: 5},
            7: {11: 3}, 8: {14: 50}, 9: {10: 3, 12: 2}, 10: {14: 10}, 11: {6: 0.3, 12: 4}, 12: {8: 50},
            13: {12: 5}, 14: {6: 0.2}, 15: {10: 2}, 16: {6: 0.4}, 17: {12: 3}, 18: {8: 30, 12: 2}, 19: {6: 0.3},
            20: {12: 1}, 21: {12: 4}, 22: {12: 6}, 23: {6: 0.2}, 24: {6: 0.1}, 25: {10: 10}, 26: {11: 5},
            27: {1: 50}, 28: {12: 7}, 29: {14: 30}, 30: {11: 1}, 31: {13: 2}, 32: {13: 1}, 33: {6: 0.3},
            34: {5: -1}, 35: {10: 15}, 36: {12: 2}
        }

        self.use_collected_states = use_collected_states
        if self.use_collected_states:
            try:
                self.collected_states = np.load("environment_states_grok.npy")
            except FileNotFoundError:
                print("Collected states file not found. Falling back to default state.")
                self.collected_states = None
        else:
            self.collected_states = None

    def normalize_state(self, state_sequence):
        """
        Normalize the entire state sequence to the range [0, 1].

        Args:
            state_sequence (np.ndarray): State sequence of shape (5, 25).

        Returns:
            np.ndarray: Normalized state sequence of shape (5, 25).
        """
        state_sequence = np.clip(state_sequence, self.state_min, self.state_max)
        normalized_sequence = (state_sequence - self.state_min) / (self.state_max - self.state_min + 1e-8)
        return normalized_sequence

    def reset(self):
        """
        Reset the environment to the initial state.

        Returns:
            np.ndarray: Normalized initial state sequence of shape (5, 25).
        """
        if self.use_collected_states and self.collected_states is not None:
            idx = random.randint(0, len(self.collected_states) - 1)
            self.state = np.clip(self.collected_states[idx], self.state_min, self.state_max)
        else:
            # Default state with realistic values aligned with state_max
            self.state = np.array([2, 2, 50.0, 1, 1, 100, 50.0, 1, 500, 0.5, 2, 500, 0, 1000.0, 2, 5000, 25.0, 1, 50, 0, 0, 0, 5000, 7500, 1.0], dtype=np.float32)
        
        self.state_history.clear()
        for _ in range(5):
            self.state_history.append(self.state.copy())
        
        state_sequence = np.array(self.state_history)
        return self.normalize_state(state_sequence)

    def step(self, action):
        """
        Take a step in the environment based on the action.

        Args:
            action (int): The action to take (0 to 36).

        Returns:
            tuple: (observation, reward, done, info)
        """
        for idx, delta in self.chaos_action_effects.get(action, {}).items():
            self.state[idx] = max(0, self.state[idx] + delta)
        self.state = np.clip(self.state, self.state_min, self.state_max)

        self.state_history.append(self.state.copy())
        if len(self.state_history) > 5:
            self.state_history.popleft()
        state_sequence = np.array(self.state_history)
        normalized_state = self.normalize_state(state_sequence)

        anomaly_score = self.predictive_model.get_anomaly_score(normalized_state)
        #print(f"action:{action},Anomaly score: {anomaly_score}")
        business_impact = max(0, 2 - self.state[1])  # Use self.state[1] for ec2_running
        reward = anomaly_score - (0.1 * self.action_costs[action]) - (0.1 * business_impact)
        #print(f"reward: {reward}")
        done = self.state[1] <= 0 or len(self.state_history) > 100  # Use self.state[1] for ec2_running

        #with open("chaos_log.txt", "a") as f:
        #    f.write(f"Action {action}, State {self.state}, Reward {reward}, Anomaly Score {anomaly_score}\n")

        return normalized_state, reward, done, {}


        '''
        try:
            # Perform the chaos action (AWS operations)
            if action == 0 and self.instance_ids and self.instance_ids[0] != 'i-123':
                self.ec2.terminate_instances(InstanceIds=[self.instance_ids[0]])
            elif action == 1:
                self.s3.put_bucket_policy(
                    Bucket='sensitive-data-bucket',
                    Policy='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":"*","Action":"s3:GetObject","Resource":"arn:aws:s3:::sensitive-data-bucket/*"}]}'
                )
            elif action == 2:
                self.ec2.stop_instances(InstanceIds=[self.instance_ids[0]])
            elif action == 3:
                self.rds.stop_db_instance(DBInstanceIdentifier='chaos-db')
            elif action == 4:
                self.s3.delete_bucket(Bucket='sensitive-data-bucket')
            elif action == 5:
                self.dynamodb.delete_table(TableName='ChaosTable')
            elif action == 6:
                self.lambda_.update_function_configuration(FunctionName='ChaosFunction', Timeout=10)
            elif action == 7:
                self.elb.delete_load_balancer(LoadBalancerName='chaos-elb')
            elif action == 8:
                self.apigateway.delete_rest_api(RestApiId=self.api_id)
            elif action == 9:
                self.ec2.reboot_instances(InstanceIds=[self.instance_ids[0]])
            elif action == 10:
                self.s3.put_bucket_cors(
                    Bucket='sensitive-data-bucket',
                    CORSConfiguration={'CORSRules': [{'AllowedMethods': ['GET'], 'AllowedOrigins': ['*']}]}
                )
            elif action == 11:
                self.lambda_.update_function_configuration(FunctionName='ChaosFunction', MemorySize=256)
            elif action == 12:
                self.dynamodb.update_table(
                    TableName='ChaosTable',
                    ProvisionedThroughput={'ReadCapacityUnits': 1, 'WriteCapacityUnits': 1}
                )
            elif action == 13:
                self.rds.reboot_db_instance(DBInstanceIdentifier='chaos-db')
            elif action == 14:
                self.lambda_.update_function_configuration(FunctionName='ChaosFunction', Timeout=5)
            elif action == 15:
                self.ec2.modify_instance_attribute(InstanceId=self.instance_ids[0], Attribute='instanceType', Value='t2.small')
            elif action == 16:
                self.lambda_.update_function_configuration(FunctionName='ChaosFunction', MemorySize=128)
            elif action == 17:
                self.dynamodb.update_table(
                    TableName='ChaosTable',
                    ProvisionedThroughput={'ReadCapacityUnits': 10, 'WriteCapacityUnits': 10}
                )
            elif action == 18:
                self.s3.delete_objects(Bucket='sensitive-data-bucket', Delete={'Objects': [{'Key': 'test'}]})
            elif action == 19:
                self.lambda_.update_function_configuration(FunctionName='ChaosFunction', Timeout=15)
            elif action == 20:
                self.dynamodb.update_table(
                    TableName='ChaosTable',
                    ProvisionedThroughput={'ReadCapacityUnits': 2, 'WriteCapacityUnits': 2}
                )
            elif action == 21:
                self.dynamodb.update_table(
                    TableName='ChaosTable',
                    ProvisionedThroughput={'ReadCapacityUnits': 15, 'WriteCapacityUnits': 15}
                )
            elif action == 22:
                self.dynamodb.update_table(
                    TableName='ChaosTable',
                    ProvisionedThroughput={'ReadCapacityUnits': 20, 'WriteCapacityUnits': 20}
                )
            elif action == 23:
                self.lambda_.update_function_configuration(FunctionName='ChaosFunction', Timeout=3)
            elif action == 24:
                self.lambda_.update_function_configuration(FunctionName='ChaosFunction', MemorySize=512)
            elif action == 25:
                self.ec2.modify_instance_attribute(InstanceId=self.instance_ids[0], Attribute='instanceType', Value='t2.micro')
            elif action == 26:
                self.elb.configure_health_check(
                    LoadBalancerName='chaos-elb',
                    HealthCheck={'Target': 'HTTP:80/', 'Interval': 5, 'Timeout': 2}
                )
            elif action == 27:
                self.s3.put_bucket_website(
                    Bucket='sensitive-data-bucket',
                    WebsiteConfiguration={'IndexDocument': {'Suffix': 'index.html'}}
                )
            elif action == 28:
                self.dynamodb.update_table(
                    TableName='ChaosTable',
                    ProvisionedThroughput={'ReadCapacityUnits': 25, 'WriteCapacityUnits': 25}
                )
            elif action == 29:
                self.apigateway.update_rest_api(RestApiId=self.api_id, PatchOperations=[{'op': 'replace', 'path': '/timeout', 'value': '50'}])
            elif action == 30:
                self.elb.set_load_balancer_policies_of_listener(
                    LoadBalancerName='chaos-elb',
                    LoadBalancerPort=80,
                    PolicyNames=['ELBSecurityPolicy-2016-08']
                )
            elif action == 31:
                self.cloudtrail.start_logging(Name='chaos-trail')
            elif action == 32:
                self.cloudtrail.stop_logging(Name='chaos-trail')
            elif action == 33:
                self.lambda_.update_function_configuration(FunctionName='ChaosFunction', MemorySize=1024)
            elif action == 34:
                self.rds.delete_db_instance(DBInstanceIdentifier='chaos-db', SkipFinalSnapshot=True)
            elif action == 35:
                self.ec2.modify_instance_attribute(InstanceId=self.instance_ids[0], Attribute='instanceType', Value='t2.large')
            elif action == 36:
                self.dynamodb.update_table(
                    TableName='ChaosTable',
                    ProvisionedThroughput={'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
                )

            for idx, delta in self.chaos_action_effects.get(action, {}).items():
                self.state[idx] = max(0, self.state[idx] + delta)
            self.state = np.clip(self.state, self.state_min, self.state_max)

            self.state_history.append(self.state.copy())
            if len(self.state_history) > 5:
                self.state_history.popleft()
            state_sequence = np.array(self.state_history)
            normalized_state = self.normalize_state(state_sequence)

            anomaly_score = self.predictive_model.get_anomaly_score(normalized_state)
            print(f"action:{action},Anomaly score: {anomaly_score}")
            business_impact = max(0, 2 - self.state[1])  # Use self.state[1] for ec2_running
            reward = anomaly_score - (0.1 * self.action_costs[action]) - (0.1 * business_impact)
            print(f"reward: {reward}")
            done = self.state[1] <= 0 or len(self.state_history) > 100  # Use self.state[1] for ec2_running

            #with open("chaos_log.txt", "a") as f:
            #    f.write(f"Action {action}, State {self.state}, Reward {reward}, Anomaly Score {anomaly_score}\n")

            return normalized_state, reward, done, {}

        except Exception as e:
            #print(f"Chaos action {action} failed: {e}")
            # Update state even if AWS operation fails
            for idx, delta in self.chaos_action_effects.get(action, {}).items():
                self.state[idx] = max(0, self.state[idx] + delta)
            self.state = np.clip(self.state, self.state_min, self.state_max)

            self.state_history.append(self.state.copy())
            if len(self.state_history) > 5:
                self.state_history.popleft()
            state_sequence = np.array(self.state_history)
            normalized_state = self.normalize_state(state_sequence)

            anomaly_score = self.predictive_model.get_anomaly_score(normalized_state)
            print(f"action:{action},Anomaly score: {anomaly_score}")
            business_impact = max(0, 2 - self.state[1])  # Use self.state[1] for ec2_running
            reward = anomaly_score - (0.1 * self.action_costs[action]) - (0.1 * business_impact)
            print(f"reward: {reward}")
            done = self.state[1] <= 0 or len(self.state_history) > 100  # Use self.state[1] for ec2_running

            #with open("chaos_log.txt", "a") as f:
            #    f.write(f"Action {action}, State {self.state}, Reward {reward}, Anomaly Score {anomaly_score}\n")

            return normalized_state, reward, done, {}
            '''
    def select_action(self, state):
        """
        Select an action based on the current state using the DQN model.

        Args:
            state (np.ndarray): The current state sequence.

        Returns:
            int: The selected action.
        """
        if self.model is None:
            return self.action_space.sample()
        else:
            action, _ = self.model.predict(state, deterministic=True)
            return action

def train_chaos_agent(is_local=True, use_collected_states=True):
    """
    Train the chaos agent using DQN.

    Args:
        is_local (bool): Whether to use LocalStack.
        use_collected_states (bool): Whether to use pre-collected states.

    Returns:
        DQN: The trained DQN model.
    """
    env = ChaosEnvironment(is_local=is_local, use_collected_states=use_collected_states)
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, exploration_fraction=0.3)
    model.learn(total_timesteps=10000)
    model.save("chaos_agent")
    return model

if __name__ == "__main__":
    train_chaos_agent(is_local=True, use_collected_states=True)