from numpy.random import uniform, seed
from numpy import floor
from collections import namedtuple
import json
from IPython.display import display,Javascript
import numpy as np


class AirPrice():

    def __init__(self, real_min_demand_level = 1500, real_max_demand_level = 3500, num_tickets=200, max_days=70):

        self.n_demand_levels = 11
        self.scale_min_demand_level = 100
        self.scale_max_demand_level = 200
        self.scale_max_tickets = 101
        
        max_days = max_days + 1
        num_tickets = num_tickets + 1

        self.min_demand_level = self.scale_min_demand_level
        self.max_demand_level = self.scale_max_demand_level
        self.max_tickets = self.scale_max_tickets
        min_demand_level = self.scale_min_demand_level
        max_demand_level = self.scale_max_demand_level
        max_tickets = self.scale_max_tickets

        self.real_min_demand_level = real_min_demand_level
        self.real_max_demand_level = real_max_demand_level
        self.demand_levels = np.linspace(min_demand_level, max_demand_level, self.n_demand_levels)

        max_tickets = self.max_tickets
        self.num_tickets = num_tickets
        self.max_days = max_days


        # Q function parameters are: 
        #n_sold in day, tickets_left to start day, demand_level, days_left
        self.Q = np.zeros([max_tickets, max_tickets, self.n_demand_levels, max_days])
        # V func parameters are: n_left and n_days
        self.V = np.zeros([max_tickets, max_days])

        for tickets_left in range(max_tickets):
            for tickets_sold in range(tickets_left+1): # add 1 to offset 0 indexing. Allow selling all tickets
                for demand_index, demand_level in enumerate(self.demand_levels):
                    # Zerro indicate never set zero prices
                    price = max(demand_level - tickets_sold, 0)
                    self.Q[tickets_sold, tickets_left, demand_index, 0] = price * tickets_sold
            # revenue for optimal quantity at every demand level 
            revenue_from_best_quantity_at_each_demand_level = self.Q[:, tickets_left, :, 0].max(axis=0)
            # take the average, since we don't know demand level ahead of time and all are equally likely
            self.V[tickets_left, 0] = revenue_from_best_quantity_at_each_demand_level.mean()

        for days_left in range(1, max_days):
            for tickets_left in range(max_tickets):
                for tickets_sold in range(tickets_left):
                    for demand_index, demand_level in enumerate(self.demand_levels):
                        price = max(demand_level - tickets_sold, 0)
                        rev_today = price * tickets_sold
                        self.Q[tickets_sold, tickets_left, demand_index, days_left] = rev_today + self.V[tickets_left-tickets_sold, days_left-1]
                expected_total_rev_from_best_quantity_at_each_demand_level = self.Q[:, tickets_left, :, days_left].max(axis=0)
                self.V[tickets_left, days_left] = expected_total_rev_from_best_quantity_at_each_demand_level.mean()


    def price_to_realprice(self, price):
        return (price-self.scale_min_demand_level)/(self.scale_max_demand_level-self.scale_min_demand_level)*(self.real_max_demand_level-self.real_min_demand_level)+self.real_min_demand_level

    def realprice_to_price(self, realprice):
        return (realprice-self.real_min_demand_level)/(self.real_max_demand_level-self.real_min_demand_level)*(self.scale_max_demand_level-self.scale_min_demand_level)+self.scale_min_demand_level

    def _tickets_sold(self, p, demand_level, max_qty):
        quantity_demanded = floor(max(0, p - demand_level))
        return min(quantity_demanded, max_qty)

    def simulate_revenue(self, days_left, tickets_left, pricing_function, demand_level_min, demand_level_max, rev_to_date=0, verbose=False):
        if (days_left == 0) or (tickets_left == 0):
            if verbose:
                if (days_left == 0):
                    print("The flight took off today. ")
                if (tickets_left == 0):
                    print("This flight is booked full.")
                print("Total Revenue: ${:.0f}".format(rev_to_date))
            return rev_to_date
        else:
            demand_level = uniform(demand_level_min, demand_level_max)
            p = pricing_function(days_left, tickets_left, demand_level)
            q = self._tickets_sold(demand_level, self.realprice_to_price(p), tickets_left)*self.num_tickets/self.scale_max_tickets
            if verbose:
                print("{:.0f} days before flight: "
                    "Started with {:.0f} seats. "
                    "Demand level: {:.0f}. "
                    "Price set to ${:.0f}. "
                    "Sold {:.0f} tickets. "
                    "Daily revenue is {:.0f}. Total revenue-to-date is {:.0f}. "
                    "{:.0f} seats remaining".format(days_left, tickets_left, self.price_to_realprice(demand_level), p, q, p*q, p*q+rev_to_date, tickets_left-q))
            return self.simulate_revenue(days_left = days_left-1,
                                tickets_left = tickets_left-q,
                                pricing_function=pricing_function,
                                rev_to_date=rev_to_date + p * q,
                                demand_level_min=demand_level_min,
                                demand_level_max=demand_level_max,
                                verbose=verbose)


    
    def pricing_function(self, days_left, tickets_left, demand_level):
        demand_level_index = np.abs(demand_level - self.demand_levels).argmin()
        day_index = days_left - 1 # arrays are 0 indexed
        tickets_index = int(tickets_left*self.max_tickets/self.num_tickets)  # in case it comes in as float, but need to index with it
        relevant_Q_vals = self.Q[:, tickets_index, demand_level_index, day_index]
        desired_quantity = relevant_Q_vals.argmax()# offset 0 indexing
        price = demand_level - desired_quantity
        return self.price_to_realprice(price)

    def get_price(self, days_left, tickets_left):
        demand_level = (self.min_demand_level+self.max_demand_level)/2
        q = self.pricing_function(days_left, tickets_left, demand_level)
        return q

if __name__ == "__main__":
    A = AirPrice(max_days=14,num_tickets=205)
    print(A.get_price(days_left=3, tickets_left=150))
    '''
    A = AirPrice()
    seed(0)
    Scenario = namedtuple('Scenario', 'n_days n_tickets')
    scenarios = [Scenario(n_days=70, n_tickets=200)]
    scenario_scores = []
    sims_per_scenario = 1
    for s in scenarios:
        scenario_score = sum(A.simulate_revenue(s.n_days, s.n_tickets, A.pricing_function, demand_level_min=A.min_demand_level, demand_level_max=A.max_demand_level, verbose=True)
                                    for _ in range(sims_per_scenario)) / sims_per_scenario
        print("Ran {:.0f} flights starting {:.0f} days before flight with {:.0f} tickets. "
            "Average revenue: ${:.0f}".format(sims_per_scenario,
                                                s.n_days,
                                                s.n_tickets,
                                                scenario_score))
        scenario_scores.append(scenario_score)
    score = sum(scenario_scores) / len(scenario_scores)
    print("Average revenue across all flights is ${:.0f}".format(score))
    '''