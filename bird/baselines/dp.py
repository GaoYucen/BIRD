import numpy as np


class AirPrice:
    def __init__(self, real_min_demand_level=1500, real_max_demand_level=3500, num_tickets=200, max_days=70):
        self.n_demand_levels = 11
        self.scale_min_demand_level = 100
        self.scale_max_demand_level = 200
        self.scale_max_tickets = 101
        max_days = max_days + 1
        num_tickets = num_tickets + 1
        self.min_demand_level = self.scale_min_demand_level
        self.max_demand_level = self.scale_max_demand_level
        self.max_tickets = self.scale_max_tickets
        self.real_min_demand_level = real_min_demand_level
        self.real_max_demand_level = real_max_demand_level
        self.demand_levels = np.linspace(self.min_demand_level, self.max_demand_level, self.n_demand_levels)
        self.num_tickets = num_tickets
        self.max_days = max_days
        self.Q = np.zeros([self.max_tickets, self.max_tickets, self.n_demand_levels, max_days])
        self.V = np.zeros([self.max_tickets, max_days])
        for tickets_left in range(self.max_tickets):
            for tickets_sold in range(tickets_left + 1):
                for demand_index, demand_level in enumerate(self.demand_levels):
                    price = max(demand_level - tickets_sold, 0)
                    self.Q[tickets_sold, tickets_left, demand_index, 0] = price * tickets_sold
            self.V[tickets_left, 0] = self.Q[:, tickets_left, :, 0].max(axis=0).mean()
        for days_left in range(1, max_days):
            for tickets_left in range(self.max_tickets):
                for tickets_sold in range(tickets_left):
                    for demand_index, demand_level in enumerate(self.demand_levels):
                        price = max(demand_level - tickets_sold, 0)
                        rev_today = price * tickets_sold
                        self.Q[tickets_sold, tickets_left, demand_index, days_left] = rev_today + self.V[tickets_left - tickets_sold, days_left - 1]
                self.V[tickets_left, days_left] = self.Q[:, tickets_left, :, days_left].max(axis=0).mean()

    def price_to_realprice(self, price):
        return (price-self.scale_min_demand_level)/(self.scale_max_demand_level-self.scale_min_demand_level)*(self.real_max_demand_level-self.real_min_demand_level)+self.real_min_demand_level

    def pricing_function(self, days_left, tickets_left, demand_level):
        demand_level_index = np.abs(demand_level - self.demand_levels).argmin()
        day_index = days_left - 1
        tickets_index = int(tickets_left*self.max_tickets/self.num_tickets)
        tickets_index = min(max(tickets_index, 0), self.max_tickets - 1)
        relevant = self.Q[:, tickets_index, demand_level_index, day_index]
        desired_quantity = relevant.argmax()
        price = demand_level - desired_quantity
        return self.price_to_realprice(price)

    def get_price(self, days_left, tickets_left):
        demand_level = (self.min_demand_level+self.max_demand_level)/2
        return self.pricing_function(days_left, tickets_left, demand_level)
