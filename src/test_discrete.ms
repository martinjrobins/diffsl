inputs = [p, g]

p_i { (1:3): 1 }
g { 1 }

A_ij {
    (0..1, 0..1): z = 1,
    (1, 1): 1,
}

B_ij {
    1 / p * A_ij
}

u_i {
    x,
    y,
}

dudt_i {
    dxdt
    dydt,
}

RHS_i { 
    p * 2 + g * x,
    y,
}

RHS_AUG_i {
    B_ij * RHS_i
}

F_i {
    dxdt,
    0,
}

G_i {
    RHS,
    x + y + t,
}