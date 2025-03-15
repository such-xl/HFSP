result = []
for i in range(5):
    if self._status == JobStatus.IDLE:
        result.append(cp_dict.get(i + 1, 0))
    else:
        if i + 1 == self.machine.id:
            result.append(cp_dict.get(self.machine.id))
        else:
            result.append(0)