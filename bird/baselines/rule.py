def get_price(volume, remaining_rounds, capacity, price, upper_bound, lower_bound, total_rounds):
    p = price * (1 + ((volume - capacity/total_rounds)/(capacity/total_rounds) * ((70-remaining_rounds)/70)))
    return min(max(p, lower_bound), upper_bound)
