import time
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from transformers import pipeline
from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server
from chaos_agent_grok import ChaosEnvironment
from remediation_agent_grok import RemediationEnvironment

SLACK_CHANNEL = "#chaos-engineering"
PROMETHEUS_PORT = 8000
GRAFANA_DASHBOARD_URL = "http://localhost:3000"
APPROVAL_TIMEOUT = 3600

#summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def send_slack_message(slack_client, message):
    try:
        response = slack_client.chat_postMessage(channel=SLACK_CHANNEL, text=message)
        return response["ts"]
    except SlackApiError as e:
        print(f"Error sending Slack message: {e}")
        return None

def request_approval(slack_client):
    message = "Chaos and remediation simulation is ready to start. Reply 'approve' to proceed or 'deny' to cancel."
    return send_slack_message(slack_client, message)

def check_for_approval(slack_client, original_ts):
    try:
        history = slack_client.conversations_history(channel=SLACK_CHANNEL, oldest=original_ts)
        for msg in history["messages"]:
            if msg["ts"] > original_ts and "text" in msg:
                text = msg["text"].lower()
                if "approve" in text:
                    return "approved"
                elif "deny" in text:
                    return "denied"
        return None
    except SlackApiError as e:
        print(f"Error checking Slack history: {e}")
        return None

def summarize_logs(log_file):
    try:
        with open(log_file, 'r') as f:
            logs = f.read()
        if not logs:
            return "No log data to summarize."
        chunk_size = 1000
        summaries = []
        for i in range(0, len(logs), chunk_size):
            chunk = logs[i:i + chunk_size]
            summary = summarizer(chunk, max_length=50, min_length=20, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        return " ".join(summaries)
    except FileNotFoundError:
        print(f"Log file {log_file} not found.")
        return "Log file not found."
    except Exception as e:
        print(f"Error summarizing logs: {e}")
        return "Summary could not be generated."

def run_chaos_simulation(slack_token, is_local=True, use_collected_states=True):
    
    #slack_client = WebClient(token=slack_token)
    chaos_env = ChaosEnvironment(is_local=is_local, use_collected_states=use_collected_states, load_model=True)
    remediation_env = RemediationEnvironment(is_local=is_local, use_collected_states=use_collected_states, load_model=True)
    
    registry = CollectorRegistry()
    chaos_actions_total = Counter('chaos_actions_total', 'Total chaos actions performed', ['action'], registry=registry)
    remediation_actions_total = Counter('remediation_actions_total', 'Total remediation actions performed', ['action'], registry=registry)
    latest_chaos_reward = Gauge('latest_chaos_reward', 'Latest reward from chaos action', registry=registry)
    latest_remediation_reward = Gauge('latest_remediation_reward', 'Latest reward from remediation action', registry=registry)
    '''
    start_http_server(PROMETHEUS_PORT, registry=registry)
    print(f"Prometheus metrics server started on port {PROMETHEUS_PORT}")

    approval_ts = request_approval(slack_client)
    if not approval_ts:
        print("Failed to send approval request. Exiting.")
        return

    print("Waiting for approval via Slack...")
    approval_status = None
    start_time = time.time()
    while approval_status is None and (time.time() - start_time) < APPROVAL_TIMEOUT:
        approval_status = check_for_approval(slack_client, approval_ts)
        time.sleep(10)

    if approval_status != "approved":
        print("Simulation not approved or timed out. Exiting.")
        send_slack_message(slack_client, "Simulation not approved or timed out. System stopped.")
        return

    print("Approval received. Starting simulation.")
    send_slack_message(slack_client, f"Starting simulation. Metrics at http://localhost:{PROMETHEUS_PORT}")
    '''
    state = chaos_env.reset()
    print("Initial state shape:", state.shape)
    for _ in range(10):
        action = chaos_env.select_action(state)
        next_state, reward, done, _ = chaos_env.step(int(action.item()))
        print("next_state shape:", next_state.shape)
        chaos_actions_total.labels(action=str(action)).inc()
        latest_chaos_reward.set(reward)
        print(f"Chaos action {action} performed, Reward: {reward}")
        #send_slack_message(slack_client, f"Chaos action {action} performed, Reward: {reward}")

        anomaly_score = remediation_env.predictive_model.get_anomaly_score(next_state)
        print(f"Anomaly score: {anomaly_score}")
        if anomaly_score > 0.1:
            remediation_action = remediation_env.select_action(next_state)
            mitigated_state, remediation_reward, remediation_done, _ = remediation_env.step(int(remediation_action.item()))
            remediation_actions_total.labels(action=str(remediation_action)).inc()
            latest_remediation_reward.set(remediation_reward)
            state = mitigated_state
            print(f"Remediation action {remediation_action} performed, Reward: {remediation_reward}")
            #send_slack_message(slack_client, f"Remediation action {remediation_action} performed (AS: {anomaly_score}, Reward: {remediation_reward})")
        else:
            state = next_state
            print("No remediation needed")
            #send_slack_message(slack_client, f"No remediation needed (AS: {anomaly_score})")

        if done:
            state = chaos_env.reset()
        time.sleep(60)

    chaos_summary = summarize_logs("chaos_log.txt")
    #remediation_summary = summarize_logs("remediation_log.txt")
    summary_msg = f"Chaos Summary: {chaos_summary}\nRemediation Summary: {remediation_summary}"
    print(summary_msg)
    #send_slack_message(slack_client, summary_msg)
    #send_slack_message(slack_client, "Chaos and remediation simulation completed.")

if __name__ == "__main__":
    slack_token = "xoxb-your-slack-token"  # Replace with your Slack bot token
    run_chaos_simulation(slack_token, is_local=True, use_collected_states=True)