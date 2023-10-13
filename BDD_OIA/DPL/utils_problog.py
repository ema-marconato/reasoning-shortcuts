import torch
from itertools import product

def build_world_queries_matrix_FS():

    possible_worlds = list(product(range(2), repeat=6))
    n_worlds  = len(possible_worlds)
    n_queries = 4
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries)  # (100, 20)
    for w in range(n_worlds):
        tl_green, follow, clear, tl_red, t_sign, obs = look_up[w]
        
        if tl_green + follow + clear > 0:
            if tl_green + tl_red == 2 or follow + obs == 2:
                pass
            elif tl_red + t_sign + obs > 0:
                w_q[w, 0] = 1 # not move
                w_q[w, 3] = 1 # stop
            else:
                w_q[w, 1] = 1 # move forward
                w_q[w, 2] = 1 # no-stop
        else:
            w_q[w, 0] = 1 # not move
            if tl_red + t_sign + obs > 0:
                w_q[w, 3] = 1 # stop
            else:
                w_q[w,2] = 1 # no-stop
    return w_q

def build_world_queries_matrix_LR():

    possible_worlds = list(product(range(2), repeat=7))
    n_worlds  = len(possible_worlds)
    n_queries = 2
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries)  # (100, 20)
    for w in range(n_worlds):
        tl_red, no_left_lane, left_solid_line, obs, left_lane, tl_green, follow = look_up[w]
        
        if left_lane+tl_green+follow > 0:
            if tl_green + tl_red == 2 or no_left_lane == 1:
                pass # invalid
            elif tl_red + obs + left_solid_line > 0:
                w_q[w, 0] = 1 # not move
            else:
                w_q[w, 1] = 1 # move forward
        else:
            w_q[w, 0] = 1 # not move
    # for i in range(n_worlds):
    #     print(look_up[i], '->',w_q[i])
    return w_q

def build_world_queries_matrix_L():

    possible_worlds = list(product(range(2), repeat=6))
    n_worlds  = len(possible_worlds)
    n_queries = 2
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries) 
    for w in range(n_worlds):
        left_lane, tl_green, follow, no_left_lane, obs, left_solid_line = look_up[w]

        if left_lane + tl_green + follow + no_left_lane + obs + left_solid_line == 0:
            w_q[w,0] = 0.5
            w_q[w,1] = 0.5
        elif left_lane+tl_green+follow > 0:
            w_q[w,1] = 1
        else:
            w_q[w,0] = 1
    return w_q

def build_world_queries_matrix_R():

    possible_worlds = list(product(range(2), repeat=6))
    n_worlds  = len(possible_worlds)
    n_queries = 2
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries) 
    for w in range(n_worlds):
        right_lane, tl_green, follow, no_right_lane, obs, right_solid_line = look_up[w]

        if right_lane + tl_green + follow + no_right_lane + obs + right_solid_line == 0:
            w_q[w,0] = 0.5
            w_q[w,1] = 0.5
        elif right_lane+tl_green+follow > 0:
            if obs + right_solid_line + no_right_lane > 0:
                w_q[w, 0] = 1 # not move
            else:
                w_q[w, 1] = 1 # move 
        else:
            w_q[w, 0] = 1 # not move
    return w_q

def compute_logic_forward(or_three_bits, concepts:torch.Tensor): 
    A = concepts[:,  :2].unsqueeze(2).unsqueeze(3)
    B = concepts[:, 2:4].unsqueeze(1).unsqueeze(3)
    C = concepts[:, 4:6].unsqueeze(1).unsqueeze(2)

    poss_worlds = A.multiply(B).multiply(C).view(-1, 2*2*2)

    active = torch.einsum('bi,ik->bk', poss_worlds, or_three_bits)

    # assert torch.abs(active.sum() / len(active)- 1) < 0.001, (active, active.sum() / len(active) )

    return active

def compute_logic_stop(or_six_bits, concepts:torch.Tensor): 
    A = concepts[:,  6:8 ].unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
    B = concepts[:,  8:10].unsqueeze(1).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
    C = concepts[:, 10:12].unsqueeze(1).unsqueeze(2).unsqueeze(4).unsqueeze(5).unsqueeze(6)
    D = concepts[:, 12:14].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(5).unsqueeze(6)
    E = concepts[:, 14:16].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(6)
    F = concepts[:, 16:18].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)

    poss_worlds = A.multiply(B).multiply(C).multiply(D).multiply(E).multiply(F).view(-1, 64)

    active = torch.einsum('bi,ik->bk', poss_worlds, or_six_bits)

    return active

def compute_logic_no_left(or_three_bits, concepts:torch.Tensor): 
    A = concepts[:, 18:20].unsqueeze(2).unsqueeze(3)
    B = concepts[:, 20:22].unsqueeze(1).unsqueeze(3)
    C = concepts[:, 22:24].unsqueeze(1).unsqueeze(2)

    poss_worlds = A.multiply(B).multiply(C).view(-1, 8)

    active = torch.einsum('bi,ik->bk', poss_worlds, or_three_bits)

    return active

def compute_logic_left(or_three_bits, concepts:torch.Tensor): 
    A = concepts[:, 24:26].unsqueeze(2).unsqueeze(3)
    B = concepts[:, 26:28].unsqueeze(1).unsqueeze(3)
    C = concepts[:, 28:30].unsqueeze(1).unsqueeze(2)

    poss_worlds = A.multiply(B).multiply(C).view(-1, 8)

    active = torch.einsum('bi,ik->bk', poss_worlds, or_three_bits)

    return active

def compute_logic_no_right(or_three_bits, concepts:torch.Tensor): 
    A = concepts[:, 30:32].unsqueeze(2).unsqueeze(3)
    B = concepts[:, 32:34].unsqueeze(1).unsqueeze(3)
    C = concepts[:, 34:36].unsqueeze(1).unsqueeze(2)

    poss_worlds = A.multiply(B).multiply(C).view(-1, 8)

    active = torch.einsum('bi,ik->bk', poss_worlds, or_three_bits)

    return active

def compute_logic_right(or_three_bits, concepts:torch.Tensor): 
    A = concepts[:, 36:38].unsqueeze(2).unsqueeze(3)
    B = concepts[:, 38:40].unsqueeze(1).unsqueeze(3)
    C = concepts[:, 40:42].unsqueeze(1).unsqueeze(2)

    poss_worlds = A.multiply(B).multiply(C).view(-1, 8)

    active = torch.einsum('bi,ik->bk', poss_worlds, or_three_bits)

    return active

def compute_logic_obstacle(or_four_bits, pC: torch.Tensor):
    
    clean    = pC[:, 4:6]

    o_car    = pC[:, 10:12].unsqueeze(2).unsqueeze(3).unsqueeze(4)
    o_person = pC[:, 12:14].unsqueeze(1).unsqueeze(3).unsqueeze(4)
    o_rider  = pC[:, 14:16].unsqueeze(1).unsqueeze(2).unsqueeze(4)
    o_other  = pC[:, 16:18].unsqueeze(1).unsqueeze(2).unsqueeze(3)

    obs_worlds = o_car.multiply(o_person).multiply(o_rider).multiply(o_other).view(-1, 16)

    obs_active = torch.einsum('bi,ik->bk', obs_worlds, or_four_bits)





    pass
