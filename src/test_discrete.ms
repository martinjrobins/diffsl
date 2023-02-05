A_ij {
    "filename.mat"
}

B_ij {
    1 / p * A_ij
}

u_i {
    x,
    y,
}

RHS_i { 
    p * 2 + g * x,
    y,
}

RHS_AUG_i {
    B_ij * RHS_i
}

F_i {
    dudt,
    0,
}

G_i {
    RHS,
    x + y + t,
}