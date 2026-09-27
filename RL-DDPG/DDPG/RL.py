def get_price(volume, rt, c, price, upper_bound, lower_bound, round):
    p = price*(1 + ((volume - c/round)/(c/round) * ((70-rt)/70)))
    if (p > upper_bound):
        p = upper_bound
    elif (p < lower_bound):
        p = lower_bound
    return p

