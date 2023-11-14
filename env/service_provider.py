from .task import TaskType
from .config import *


class ServiceProvider:

    def __init__(self, sid, task_type_id):
        self._sid = sid
        self._task_type = TaskType(task_type_id)
        self._reward_coefs = (
            np.random.choice(AX_RANGE),
            np.random.choice(AY_RANGE),
            np.random.choice(BX_RANGE),
            np.random.choice(BY_RANGE))
        self._serving_tasks = []
        self._terminated_tasks = {'crashed': [], 'finished': []}
        self._total_t = np.random.choice(TOTAL_T_RANGE)
        self._num_crashed = 0

        # The following info are not currently considered
        self._loc = np.random.randint(*LOCATION_RANGE, size=(1, 2))
        self._num_cpu = NUM_CPUS
        self._num_gpu = NUM_GPUS
        self._cpu_mem = CPU_MEM
        self._gpu_mem = GPU_MEM

    def _distance_to(self, user):
        return np.sqrt(np.square(self._loc - user._loc).sum())

    @property
    def id(self):
        return self._sid

    @property
    def total_t(self):
        return self._total_t

    @property
    def used_t(self):
        return sum([task.t for task in self._serving_tasks])

    @property
    def available_t(self):
        return self._total_t - self.used_t

    def is_enough(self, task):
        return self.available_t >= task.t

    @property
    def norm_total_t(self):
        max_t = TOTAL_T_RANGE[-1]
        return self._total_t / max_t

    @property
    def norm_available_t(self):
        return self.available_t / self._total_t

    def task_summary(self):
        num_serving = len(self._serving_tasks)
        num_crashed = len(self._terminated_tasks['crashed'])
        num_finished = len(self._terminated_tasks['finished'])
        crashed_total_t = sum(
            task.t for task in self._terminated_tasks['crashed']
        )
        crashed_total_reward = sum(
            self.calculate_reward(task) for task in self._terminated_tasks['crashed']
        )
        finished_total_t = sum(
            task.t for task in self._terminated_tasks['finished']
        )
        finished_total_reward = sum(
            self.calculate_reward(task) for task in self._terminated_tasks['finished']
        )
        return {
            "total": num_serving + num_crashed + num_finished,
            "serving": num_serving,
            "crashed": num_crashed,
            "finished": num_finished,
            "crashed_total_t": crashed_total_t,
            "crashed_total_reward": crashed_total_reward,
            "finished_total_t": finished_total_t,
            "finished_total_reward": finished_total_reward
        }

    def check_finished(self, curr_time):
        num_finished = 0
        for running_task_ in self._serving_tasks[:]:
            if running_task_.can_finished(curr_time):
                running_task_.set_finished()
                self._terminated_tasks['finished'].append(running_task_)
                self._serving_tasks.remove(running_task_)
                num_finished += 1
        return num_finished

    def calculate_reward(self, task):
        # reward will be delayed in practice
        return REWARD(*self._reward_coefs, task.t)

    def assign_task(self, task, curr_time):
        reward = self.calculate_reward(task)

        # No enough resources, server crashes
        if task.t > self.available_t:
            penalty = CRASH_PENALTY_COEF
            for running_task_ in self._serving_tasks:
                running_task_.crash(curr_time)
                self._terminated_tasks['crashed'].append(running_task_)
                penalty += (1 - running_task_.progress()) * CRASH_PENALTY_COEF
            self._serving_tasks.clear()
            self._num_crashed += 1
            return -penalty

        # Allocate resources for this task
        self._serving_tasks.append(task)
        
        return reward

    def reset(self):
        self._serving_tasks.clear()
        self._terminated_tasks['crashed'].clear()
        self._terminated_tasks['finished'].clear()
        self._num_crashed = 0

    @property
    def vector(self):
        # (total_t, available_t)
        return np.hstack([self.norm_total_t, self.norm_available_t])

    @property
    def info(self):
        return {
            'id': self.id,
            'task_type': self._task_type,
            'task_serving': len(self._serving_tasks),
            'task_finished': len(self._terminated_tasks['finished']),
            'task_crashed': len(self._terminated_tasks['crashed']),
            'total_t': self._total_t,
            'used_t': self.used_t,
            'available_t': self.available_t,
            'num_crashed': self._num_crashed
        }
