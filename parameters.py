# -*- coding: UTF-8 -*-
import numpy as np
import math

import job_distribution


class Parameters:
    def __init__(self):

        self.output_filename = 'data/tmp'

        self.num_epochs = 10000        # number of training epochs  将所有数据过几遍
        self.simu_len = 10             # length of the busy cycle that repeats itself 生成的作业序列的长度
        self.num_ex = 1                # number of sequences  作业序列条数

        self.output_freq = 10          # interval for output and store parameters

        self.num_seq_per_batch = 10    # number of sequences to compute baseline
        self.episode_max_length = 200  # enforcing an artificial terminal

        self.num_res = 2               # number of resources in the system  资源池的资源种类个数
        self.num_nw = 5                # maximum allowed number of work in the queue  资源池的作业队列的最大长度

        self.time_horizon = 20         # number of time steps in the graph  graph的时间片长度（垂直长度）
        self.max_job_len = 15          # maximum duration of new jobs  每个作业的作业最长持续时间
        self.res_slot = 10             # maximum number of available resource slots  graph的最大资源量（水平长度）
        self.max_job_size = 10         # maximum resource request of new work  每个作业的最大资源需求量

        self.backlog_size = 60         # backlog queue size  最大积压作业队列长度

        self.max_track_since_new = 10  # track how many time steps since last new jobs # 距离上一次接收新作业的最大时间间隔

        self.job_num_cap = 40          # maximum number of distinct colors in current work graph 当前资源池中最多存在颜色个数

        self.new_job_rate = 0.7        # lambda in new job arrival Poisson Process 新作业到达率的泊松分布

        self.discount = 1              # discount factor 折扣因子

        self.num_machines = 2          # 资源池数量

        # distribution for new job arrival  作业生成器
        self.dist = job_distribution.Dist(self.num_res, self.max_job_size, self.max_job_len)

        # graphical representation 图像表示
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = int(math.ceil(self.backlog_size / float(self.time_horizon))) # backlog的宽度
        self.network_input_height = self.time_horizon # 神经网络的输入矩阵的高度
        self.network_input_width = \
            (self.res_slot * self.num_machines +
             self.max_job_size * self.num_nw) * self.num_res + \
            self.backlog_width + \
            1  # for extra info, 1) time since last new job # 神经网络的输入矩阵的宽度【（单资源宽度+作业队列最大宽度）*资源种类个数+backlog宽度】

        # compact representation  紧凑表示
        self.network_compact_dim = (self.num_res + 1) * \
            (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_nw * self.num_machines + 1  # + 1 for void action

        self.delay_penalty = -1       # penalty for delaying things in the current work screen
        self.hold_penalty = -1        # penalty for holding things in the new work screen
        self.dismiss_penalty = -1     # penalty for missing a job because the queue is full

        self.num_frames = 1           # number of frames to combine and process
        self.lr_rate = 0.001          # learning rate 学习率 决定参数移动到最优值的速度快慢
        self.rms_rho = 0.9            # for rms prop
        self.rms_eps = 1e-9           # for rms prop

        self.unseen = False  # change random seed to generate unseen example

        # supervised learning mimic policy
        self.batch_size = 10
        self.evaluate_policy_name = "SJF"

    def compute_dependent_parameters(self):
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = self.backlog_size / self.time_horizon
        self.network_input_height = self.time_horizon
        self.network_input_width = \
            (self.res_slot * self.num_machines +
             self.max_job_size * self.num_nw) * self.num_res + \
            self.backlog_width + \
            1  # for extra info, 1) time since last new job

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
            (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_nw * self.num_machines + 1  # + 1 for void action

