import time

_NANO =  1000000000
class Timer:
    start_time : float
    end_time : float
    running : bool 
    
    def __init__(self) -> None:
        pass
    
    def start(self):
        self.start_time = time.time_ns()
        self.running = True
    
    def stop(self):
        if self.running:
            self.end_time = time.time_ns()
            self.running = False

    def reset(self):
        self.start_time = 0
        self.running = False

    def getTime(self):
        if self.running:
            return round((time.time_ns() - self.start_time)/_NANO,4)
        else:
            return round((self.end_time - self.start_time)/_NANO,4)
