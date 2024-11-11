# HFSP
Hybird flow shop scheduling problems
# new new 
# issue
理论上agent与其对应的job在各自链表中的位置是一致的，但这种处理方式缺乏agent异常处理。后续如果有需要应该补齐，
# new branch marl_fjsp
完善多智能体强化学习



# test0:
    after run()
    def reward_func_0(self,scale_factor,done):
    """
        简单返回time_step * scler_factor的负数
    """
    if done:
        return -self._time_step*scale_factor
    return 0
logs path = logs/record0.json
model path = madels/model0.path
Pearson correlation coefficient: vla20.fjs -0.9999999999999999

# test1:
    before run()
    def reward_func_1(self):
    """
        返回in_progress_jobs的相对延迟的率
    """
    in_progress_job = self._in_progress_jobs.head
    count = 0
    delay_rate = 0
    while in_progress_job:
        if in_progress_job._t_processed == 0:
            delay_rate += in_progress_job.get_delay_ratio()
            count +=1
        in_progress_job = in_progress_job.next
    return -delay_rate/count if count else 0