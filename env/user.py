from .config import *


class User:

    def __init__(self, uid, task_type_id):
        self._uid = uid
        self._task_type_id = task_type_id
        self._task = None
        self._all_tasks = []

        # The following info are not currently considered
        self._loc = np.random.randint(*LOCATION_RANGE, size=(1, 2))

    def add_task(self, task):
        task.set_task_type(self._task_type_id)
        self._task = task
        self._all_tasks.append(task)

    @property
    def task(self):
        return self._task

    @property
    def all_task(self):
        return self._all_tasks

    @property
    def vector(self):
        assert self._task, "No task assigned"
        return self._task.vector

    def reset(self):
        self._task = None
        self._all_tasks.clear()
