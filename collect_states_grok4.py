import random
import time
import numpy as np
from botocore.exceptions import ClientError
import argparse

class StateCollector:
    def __init__(self, use_localstack=False, endpoint_url=None, region='us-east-1'):
        """
        Initialize the StateCollector for generating synthetic system states.

        Args:
            use_localstack (bool): If True, use LocalStack instead of AWS.
            endpoint_url (str, optional): Custom endpoint URL for LocalStack.
            region (str): AWS region to simulate.
        """
        self.use_localstack = use_localstack
        endpoint_url = endpoint_url if endpoint_url else 'http://localhost:4566' if use_localstack else None
        kwargs = {
            'region_name': region,
            'endpoint_url': endpoint_url
        }
        if use_localstack:
            kwargs.update({
                'aws_access_key_id': 'test',
                'aws_secret_access_key': 'test'
            })
        self.scenarios = [
            'normal', 'high_load', 'resource_exhaustion',
            'network_issues', 'security_incidents', 'service_outages'
        ]
        self.states = []
        self.labels = []
        self.current_state = None
        self.current_scenario = None

    def collect_state(self, scenario=None, temporal=False):
        """
        Generate a synthetic state based on the provided scenario or a random one.

        Args:
            scenario (str, optional): Specific scenario (e.g., 'normal'). If None, choose randomly.
            temporal (bool): If True, generate state based on the previous state for temporal continuity.

        Returns:
            dict: Generated state with metrics and timestamp.
        """
        if scenario is None:
            scenario = random.choice(self.scenarios)
        if temporal and self.current_state is not None and self.current_scenario == scenario:
            state = self._generate_next_state(scenario)
        else:
            state = self._generate_independent_state(scenario)
            if temporal:
                self.current_state = state.copy()
                self.current_scenario = scenario
        state['timestamp'] = time.time()
        return state

    def _generate_independent_state(self, scenario):
        """Generate an independent state based on the scenario."""
        state = {}
        state.update(self._generate_ec2_metrics(scenario))
        state.update(self._generate_rds_metrics(scenario))
        state.update(self._generate_elb_metrics(scenario))
        state.update(self._generate_lambda_metrics(scenario))
        state.update(self._generate_s3_metrics(scenario))
        state.update(self._generate_sqs_metrics(scenario))
        state.update(self._generate_security_metrics(scenario))
        state.update(self._generate_network_metrics(scenario))
        return state

    def _generate_next_state(self, scenario):
        """Generate the next state based on the current state and scenario for temporal dynamics."""
        state = self.current_state.copy()
        if scenario == 'normal':
            state['ec2_cpu_avg'] = max(0, min(100, state['ec2_cpu_avg'] + random.uniform(-5, 5)))
            state['rds_connections'] = max(0, min(200, state['rds_connections'] + random.randint(-10, 10)))
        elif scenario == 'high_load':
            state['ec2_cpu_avg'] = min(100, state['ec2_cpu_avg'] + random.uniform(5, 10))
            state['rds_connections'] = min(200, state['rds_connections'] + random.randint(10, 20))
        elif scenario == 'resource_exhaustion':
            state['ec2_cpu_avg'] = min(100, state['ec2_cpu_avg'] + random.uniform(10, 20))
            state['lambda_errors'] = min(10, state['lambda_errors'] + random.randint(1, 3))
        elif scenario == 'network_issues':
            state['packet_loss_percent'] = min(10, state['packet_loss_percent'] + random.uniform(1, 5))
        elif scenario == 'security_incidents':
            state['failed_logins'] = min(50, state['failed_logins'] + random.randint(1, 5))
        elif scenario == 'service_outages':
            state['ec2_running'] = max(0, state['ec2_running'] - random.randint(0, 1))
        self.current_state = state
        return state

    def _generate_ec2_metrics(self, scenario):
        if scenario == 'normal':
            ec2_running = random.randint(2, 4)  # Away from 0 and 5
            ec2_cpu_avg = random.uniform(20, 40)  # Middle range
        elif scenario == 'high_load':
            ec2_running = 5  # High but allows decrease
            ec2_cpu_avg = random.uniform(60, 80)
        elif scenario == 'resource_exhaustion':
            ec2_running = 5
            ec2_cpu_avg = random.uniform(80, 95)  # Below max to allow decrease
        elif scenario == 'service_outages':
            ec2_running = random.randint(0, 2)  # Low, allows increase
            ec2_cpu_avg = random.uniform(10, 30)
        else:
            ec2_running = random.randint(1, 4)
            ec2_cpu_avg = random.uniform(20, 80)
        return {'ec2_count': 5, 'ec2_running': ec2_running, 'ec2_cpu_avg': ec2_cpu_avg}

    def _generate_rds_metrics(self, scenario):
        if scenario == 'normal':
            rds_available = 2
            rds_connections = random.randint(20, 60)  # Away from 0 and 200
            rds_cpu = random.uniform(10, 30)
        elif scenario == 'high_load':
            rds_available = 2
            rds_connections = random.randint(140, 180)  # High but below max
            rds_cpu = random.uniform(50, 70)
        else:
            rds_available = random.randint(1, 2)
            rds_connections = random.randint(20, 140)
            rds_cpu = random.uniform(10, 50)
        return {'rds_count': 2, 'rds_available': rds_available, 'rds_connections': rds_connections, 'rds_cpu': rds_cpu}

    def _generate_elb_metrics(self, scenario):
        if scenario == 'normal':
            elb_requests = random.randint(200, 600)  # Away from 100 and 2000
            elb_latency = random.uniform(0.02, 0.1)
        elif scenario == 'high_load':
            elb_requests = random.randint(1400, 1800)  # High but below max
            elb_latency = random.uniform(0.1, 0.4)
        else:
            elb_requests = random.randint(200, 1400)
            elb_latency = random.uniform(0.02, 0.3)
        return {'elb_count': 1, 'elb_requests': elb_requests, 'elb_latency': elb_latency}

    def _generate_lambda_metrics(self, scenario):
        if scenario == 'normal':
            lambda_invocations = random.randint(100, 200)  # Away from 50 and 500
            lambda_errors = random.randint(1, 3)  # Above 0, below 10
            lambda_duration = random.uniform(150, 250)
        elif scenario == 'resource_exhaustion':
            lambda_invocations = random.randint(300, 400)
            lambda_errors = random.randint(4, 8)
            lambda_duration = random.uniform(600, 900)
        else:
            lambda_invocations = random.randint(100, 300)
            lambda_errors = random.randint(1, 5)
            lambda_duration = random.uniform(150, 500)
        return {'lambda_count': 3, 'lambda_invocations': lambda_invocations, 'lambda_errors': lambda_errors, 'lambda_duration': lambda_duration}

    def _generate_s3_metrics(self, scenario):
        # S3 metrics are not scenario-dependent but adjusted to middle range
        s3_object_count = random.randint(2000, 4000)  # Away from 1000 and 5000
        s3_total_size = random.uniform(2e9, 4e9)  # Away from 1e9 and 5e9
        return {'s3_bucket_count': 10, 's3_object_count': s3_object_count, 's3_total_size': s3_total_size}

    def _generate_sqs_metrics(self, scenario):
        if scenario == 'normal':
            sqs_message_count = random.randint(10, 100)  # Away from 0 and 500
        elif scenario == 'high_load':
            sqs_message_count = random.randint(300, 450)
        else:
            sqs_message_count = random.randint(10, 300)
        return {'sqs_queue_count': 2, 'sqs_message_count': sqs_message_count}

    def _generate_security_metrics(self, scenario):
        if scenario == 'normal':
            security_findings = random.randint(1, 3)  # Above 0, below 20
            failed_logins = random.randint(1, 5)  # Above 0, below 50
            vulnerability_count = random.randint(1, 2)  # Above 0, below 10
        elif scenario == 'security_incidents':
            security_findings = random.randint(8, 15)
            failed_logins = random.randint(15, 30)
            vulnerability_count = random.randint(3, 7)
        else:
            security_findings = random.randint(1, 5)
            failed_logins = random.randint(1, 10)
            vulnerability_count = random.randint(1, 3)
        return {'security_findings': security_findings, 'failed_logins': failed_logins, 'vulnerability_count': vulnerability_count}

    def _generate_network_metrics(self, scenario):
        if scenario == 'normal':
            network_in = random.uniform(2e6, 4e6)  # Away from 1e5 and 5e6
            network_out = random.uniform(2e6, 4e6)
            packet_loss_percent = random.uniform(0.1, 1)  # Above 0, below 10
        elif scenario == 'network_issues':
            network_in = random.uniform(1e5, 1e6)
            network_out = random.uniform(1e5, 1e6)
            packet_loss_percent = random.uniform(5, 9)  # Below max
        else:
            network_in = random.uniform(1e6, 4e6)
            network_out = random.uniform(1e6, 4e6)
            packet_loss_percent = random.uniform(0.1, 5)
        return {'network_in': network_in, 'network_out': network_out, 'packet_loss_percent': packet_loss_percent}

    def collect_samples(self, num_samples=5000, interval=0, scenario=None, temporal=False, label=False, anomaly_prob=0.5):
        """
        Collect a specified number of synthetic state samples with a probability of anomalies.

        Args:
            num_samples (int): Number of samples to collect.
            interval (float): Time interval between samples in seconds.
            scenario (str, optional): Specific scenario for all states.
            temporal (bool): If True, generate temporally correlated states.
            label (bool): If True, record the scenario for each state.
            anomaly_prob (float): Probability of generating an anomalous state (default 0.5).

        Returns:
            list or tuple: List of states, or (states, labels) if label=True.
        """
        print(f"Collecting {num_samples} synthetic state samples with interval {interval}s and anomaly probability {anomaly_prob}")
        self.states = []
        self.labels = []
        for i in range(num_samples):
            if scenario is None:
                if random.random() < anomaly_prob:
                    chosen_scenario = random.choice([s for s in self.scenarios if s != 'normal'])
                else:
                    chosen_scenario = 'normal'
            else:
                chosen_scenario = scenario
            state = self.collect_state(chosen_scenario, temporal)
            self.states.append(state)
            if label:
                self.labels.append(chosen_scenario)
            print(f"Collected state {i+1}/{num_samples}")
            if i < num_samples - 1:
                time.sleep(interval)
        print(f"Collected {len(self.states)} samples")
        return (self.states, self.labels) if label else self.states

    def save_states(self, filename="environment.npy", label_filename=None):
        """
        Save collected states and optional labels to files.

        Args:
            filename (str): File to save states.
            label_filename (str, optional): File to save labels if provided.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.states:
            print("No states to save.")
            return False
        try:
            metric_keys = [
                'ec2_count', 'ec2_running', 'ec2_cpu_avg',
                'rds_count', 'rds_available', 'rds_connections', 'rds_cpu',
                'elb_count', 'elb_requests', 'elb_latency',
                'lambda_count', 'lambda_invocations', 'lambda_errors', 'lambda_duration',
                's3_bucket_count', 's3_object_count', 's3_total_size',
                'sqs_queue_count', 'sqs_message_count',
                'security_findings', 'failed_logins', 'vulnerability_count',
                'network_in', 'network_out', 'packet_loss_percent'
            ]
            states_array = np.array([[state[key] for key in metric_keys] for state in self.states])
            np.save(filename, states_array)
            print(f"Saved {len(self.states)} states to {filename}")
            if label_filename and self.labels:
                np.save(label_filename, np.array(self.labels))
                print(f"Saved labels to {label_filename}")
            return True
        except Exception as e:
            print(f"Error saving states: {str(e)}")
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic states for training.")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of state samples to generate.")
    parser.add_argument("--interval", type=float, default=0, help="Interval in seconds between collecting states.")
    parser.add_argument("--filename", type=str, default="environment_states_grok.npy", help="Filename to save the states.")
    parser.add_argument("--scenario", type=str, default=None, help="Specific scenario to generate states for.")
    parser.add_argument("--temporal", action="store_true", help="Generate temporally correlated states.")
    parser.add_argument("--label", action="store_true", help="Record scenario labels for validation.")
    parser.add_argument("--label_filename", type=str, default="labels.npy", help="Filename to save labels.")
    parser.add_argument("--anomaly_prob", type=float, default=0.5, help="Probability of generating anomalous states.")
    args = parser.parse_args()

    collector = StateCollector(use_localstack=True)
    collector.collect_samples(
        num_samples=args.num_samples,
        interval=args.interval,
        scenario=args.scenario,
        temporal=args.temporal,
        label=args.label,
        anomaly_prob=args.anomaly_prob
    )
    collector.save_states(args.filename, args.label_filename)