

def print_current_curriculum(curriculum):

    reward, landmark =curriculum

    r_val, r_desc = reward
    l_val, l_desc = landmark

    to_print=f"""Rewards:
    \t {r_val} : {r_desc}
    Landmark:
    \t {l_val} : {l_desc}
    """

    print(to_print)