import datetime
import tqdm
import configargparse
import numpy as np
from database import Config, ConfigHolder, Graph, Task, TaskJobs, get_session, AngularGraphSolution
import instance_evolver, instance_evolver_greedy, instance_tests

PROCESSOR_MAP = {
    "instance_evolver": instance_evolver,
    "instance_evolver_greedy": instance_evolver_greedy,
    "instance_test": instance_tests
}

def process_tasks(url, task_id, check_stuck, local):
    session = get_session(url)
    if check_stuck:
        print("Check for stuck tasks with more that 24 hours idle time...")
        stuck_tasks = session.query(Task).filter(Task.last_updated <= datetime.datetime.now() - datetime.timedelta(hours=24))
        for task in stuck_tasks:
            if input(f"Task {task.id} with type {task.task_type} is still processing. Reset? (y/N)").lower() in ["y", "yes"]:
                task.status = "RESTART"
        session.commit()
    if task_id is None:
        print("No specific task was given. All unprocessed tasks will be processed")
        ignore_stati = [Task.STATUS_OPTIONS.FINISHED, Task.STATUS_OPTIONS.PROCESSING, Task.STATUS_OPTIONS.ERROR, Task.STATUS_OPTIONS.ABORTED]
        tasks = session.query(Task).filter(Task.status.notin_(ignore_stati))\
            .filter(Task.parent_id == None).all()
    else:
        tasks = session.query(Task).filter(Task.id == task_id).all()
    for task in tasks:
        config = ConfigHolder(task)
        if local:
            config.local = True
        try:
            PROCESSOR_MAP[task.task_type].process_task(config, task, session)
        except InterruptedError:#Exception:
            pass
    

def main():
    parser = configargparse.ArgumentParser(description="Task solver for a given task (or all unprocessed)")
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file',
        is_config_file_arg=True)
    parser.add_argument('--task', type=int, default=None, help="ID of the task that will be processed. If none given, all unfinished tasks will be solved")
    parser.add_argument('--check-stuck-processing', action="store_true", help="Check if also stuck processing tasks should be reset")
    parser.add_argument('--url_path', type=str, default="angular.db", help="Path to database (default: angular.db")
    parser.add_argument('--local', action="store_true", help="Run tasks local if possible")
    parsed = parser.parse_args()
    process_tasks(parsed.url_path, parsed.task, parsed.check_stuck_processing, parsed.local)

if __name__ == "__main__":
    main()
