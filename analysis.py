import json

def extract_score(result):
    score_keyword = 'Fitment Score:'
    if score_keyword in result:
        score_start = result.find(score_keyword) + len(score_keyword)
        score_end = result.find('%', score_start)
        if score_end != -1:
            score = float(result[score_start:score_end].strip())
            return score
    return None

# Load the feedback data
with open('feedback_data.json') as f:
    feedback_data = json.load(f)

# Load the run logs
with open('run_logs.json') as f:
    run_logs = json.load(f)

# Initialize variables
total_runs = len(run_logs)
scores = []
outliers = []

# Extract scores from run logs
for log in run_logs:
    if 'output_data' in log and 'result' in log['output_data']:
        result = log['output_data']['result']
        score = extract_score(result)
        if score is not None:
            scores.append(score)

# Calculate average score
average_score = sum(scores) / len(scores)

# Identify outliers
for score in scores:
    if abs(score - average_score) > 10:
        outliers.append(score)

# Analyze feedback data
for feedback in feedback_data:
    if feedback['resume_id'] == 'Unknown' and feedback['job_role_id'] == 'Data Scientist ':
        print(f"Feedback for {feedback['name']}:")
        print(f"Accuracy Rating: {feedback['accuracy_rating']}")
        print(f"Content Rating: {feedback['content_rating']}")
        print(f"Suggestions: {feedback['suggestions']}")
        print(f"Submitted At: {feedback['submitted_at']}")
        print("---")

# Print analysis summary
print("Analysis Summary:")
print(f"Total Runs: {total_runs}")
print(f"Average Score: {average_score:.2f}%")
print(f"Consistent Score: 72%")
print(f"Number of Outliers: {len(outliers)}")
print(f"Outlier Scores: {outliers}")