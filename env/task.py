from .config import *


class TaskType:

    def __init__(self, type_id):
        self._num_types = NUM_TASK_TYPES
        assert type_id < self._num_types
        self._type_id = type_id

    @property
    def one_hot(self):
        return np.eye(self._num_types)[self._type_id]


class Task:

    def __init__(self, task_id, arrival_time, t):
        self._task_id = task_id
        self._arrival_time = arrival_time
        self._t = t
        self._t_range = T_RANGE
        self._task_type = None
        self._runtime = RUNTIME(self._t)
        self._crashed = False
        self._crash_time = -1
        self._finished = False

        # The following info are not currently considered
        self._num_cpu = 1
        self._num_gpu = 1
        self._cpu_mem = CPU_MEM_OCCUPY
        self._gpu_mem = GPU_MEM_OCCUPY
        self._data_type = 'image'
        self._data_bytes = IMG_BUFFER
        self._data_shape = IMG_CHW

    def set_task_type(self, type_id):
        self._task_type = TaskType(type_id)

    @property
    def arrival_time(self):
        return self._arrival_time

    @property
    def task_type(self):
        return self._task_type.one_hot

    @property
    def t(self):
        return self._t

    @property
    def norm_t(self):
        return self._t / self._t_range[-1]

    @property
    def norm_runtime(self):
        return self._runtime / RUNTIME(max(self._t_range))

    @property
    def vector(self):
        assert self._task_type is not None, \
            f"Please set task type for task {self._task_id} first"

        vec = np.hstack([self.norm_t, self.norm_runtime])
        return vec

    def can_finished(self, curr_time):
        assert not self._crashed, \
            f"This task {self._task_id} has been crashed"

        return curr_time >= self._arrival_time + self._runtime

    def set_finished(self):
        assert not self._crashed, \
            f"This task {self._task_id} has been crashed"

        self._finished = True

    @property
    def finished(self):
        return self._finished

    def crash(self, curr_time):
        assert not self._crashed, \
            f"This task {self._task_id} has been crashed"

        self._crash_time = curr_time
        self._crashed = True

    def progress(self, curr_time=None):
        if self._finished:
            return 1.

        if self._crashed:
            return (self._crash_time - self._arrival_time) / self._runtime

        assert curr_time, "Current time unknown"
        return (curr_time - self._arrival_time) / self._runtime


class TaskGenerator:

    def __init__(self):
        self._task_id_counter = 0
        self._lambda = LAMBDA
        self._total_time = TOTAL_TIME
        self._total_task = 0
        self._task_arrival_time = None
        self._t_range = T_RANGE
        self.reset()

    def reset(self):
        self._task_id_counter = 0
        self._total_task = np.random.poisson(self._lambda * self._total_time)
        self._task_arrival_time = np.hstack(
            [[0], np.sort(np.random.random(self._total_task) * self._total_time)])
        self._task_arrival_time = self._task_arrival_time.astype(np.int64)
        self._total_task = len(self._task_arrival_time)

    def __next__(self):
        task_id = self._task_id_counter
        assert task_id < self._total_task, "number of tasks out of range"

        arrival_time = self._task_arrival_time[task_id]
        required_t = np.random.choice(self._t_range)

        task = Task(task_id, arrival_time, required_t)

        self._task_id_counter += 1
        terminate = True if self._task_id_counter == self._total_task else False
        return task, terminate
