# HFSP
Hybird flow shop scheduling problems
# new new 
# issue
在env.run_a_time_step function中，按busy_agent_list和in_progress_jobs_list的顺序依次执行各自的run_a_time_step()
理论上agent与其对应的job在各自链表中的位置是一致的，但这种处理方式缺乏agent异常处理。后续如果有需要应该补齐，
还有就是agent和job的执行不太直观，后续考虑在machine类中添加一个变量指向它正在处理的job,在其run_a_time_step()中调用job的run_a_time_step(),删去其job_id类变量。
