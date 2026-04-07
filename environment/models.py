from pydantic import BaseModel
from typing import List

class WarehouseState(BaseModel):
    inventory: List[float]
    in_transit: List[List[float]]
    days_to_expiry: List[int]
    demand_history: List[List[float]]
    storage_used: float
    day_of_week: int
    supplier_reliability: List[float]

class OrderAction(BaseModel):
    order_quantities: List[int]  # one per product

class StepResult(BaseModel):
    state: WarehouseState
    reward: float          # 0.0–1.0
    done: bool
    info: dict             # fill_rate, waste, costs breakdown