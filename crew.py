# crew.py

class Crew:
    def __init__(self, tasks):
        self.tasks = tasks

    def kickoff(self):
        results = []
        for task in self.tasks:
            try:
                task_result = task.execute()
                results.append(task_result)
            except Exception as e:
                error_message = f"Task '{task.name}' failed: {str(e)}"
                results.append(error_message)
                print(error_message)  # Or use a logger for better error handling

        # Combine the results into a single string
        final_result = "\n\n".join(results)

        return final_result