class LossRecorder:
    def __init__(self):
        self.loss_list = []
        self.loss_total = 0.0

    def add(self, *, epoch, step, loss):
        if epoch == 0:
            self.loss_list.append(loss)
        else:
            while len(self.loss_list) <= step:
                self.loss_list.append(0.0)
            self.loss_total -= self.loss_list[step]
            self.loss_list[step] = loss
        self.loss_total += loss

    @property
    def moving_average(self):
        return self.loss_total / len(self.loss_list) if self.loss_list else 0.0

    def state_dict(self):
        return {"loss_list": self.loss_list.copy(), "loss_total": self.loss_total}

    def load_state_dict(self, state):
        self.loss_list = list(state["loss_list"])
        self.loss_total = state["loss_total"]
