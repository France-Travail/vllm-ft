import collections

class EncoderBudget:
    def __init__(self, total_budget: int):
        self.total_budget = total_budget
        self.current_usage = 0
        self.budget_cache = collections.OrderedDict()

    def request_budget(self, request_id: str, tokens: int) -> bool:
        if self.current_usage + tokens <= self.total_budget:
            self.current_usage += tokens
            self.budget_cache[request_id] = tokens
            return True
        return False

    def update_budget(self, request_id: str, actual_tokens: int, original_tokens: int) -> int:
        if request_id not in self.budget_cache:
            return self.current_usage

        original = self.budget_cache.pop(request_id)
        reclaimed = original - actual_tokens
        self.current_usage -= reclaimed
        return self.current_usage

    def reset(self):
        self.current_usage = 0
        self.budget_cache.clear()
